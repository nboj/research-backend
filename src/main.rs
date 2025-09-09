use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    fs::FileType,
    io::Error,
    process::Command,
};

use async_openai::{
    Client,
    types::{CreateCompletionRequestArgs, CreateImageRequestArgs, ImageResponseFormat, ImageSize},
};
use base64::{Engine, engine::general_purpose};
use dotenv::dotenv;
use rocket::{
    http::{Status, ext::IntoCollection},
    response::status,
    serde::{
        Deserialize, Serialize,
        json::{Json, serde_json::json},
    },
    tokio::fs::{create_dir_all, read, read_dir, read_to_string, remove_dir_all},
};
use uuid::Uuid;

#[macro_use]
extern crate rocket;

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct TokenMap {
    prompt: String,
    tokens_all: Vec<String>,
    mapping: Vec<MapEntry>,
}

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct MapEntry {
    word: String,
    word_idx: usize,
    token_indices: Vec<usize>,
    tokens: Vec<String>, // display tokens (e.g., ["Ġdog"])
    heatmap: String,     // e.g., "dog.heat_map.png"
}

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct GenerateProps {
    userid: String,
    seed: String,
    prompt: String,
}

#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
struct GenerateResponse {
    images: Vec<String>,
    tokens: Vec<String>,
}
// NOTE: needs
// * userid
// * prompt
#[post("/generate", data = "<data>")]
async fn generate(
    data: Json<GenerateProps>,
) -> Result<(Status, Json<GenerateResponse>), status::Custom<Json<ErrBody>>> {
    let id = Uuid::new_v4();
    match create_dir_all(format!("./data/{id}")).await {
        Ok(()) => {}
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(status::Custom(
                Status::InternalServerError,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };
    log::info!("here");
    let mut handle = match Command::new("./venv/Scripts/python")
        .arg("./src-py/maps-test.py")
        .arg(data.prompt.clone())
        .arg(data.seed.clone())
        .arg(format!("./data/{id}"))
        .spawn()
    {
        Ok(h) => h,
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(status::Custom(
                Status::InternalServerError,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };
    log::info!("now here");
    let res = match handle.wait() {
        Ok(v) => v,
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(status::Custom(
                Status::InternalServerError,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };
    log::info!("Status Code: {res}");

    let mut dir = match read_dir(format!("./data/{id}")).await {
        Ok(d) => d,
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(status::Custom(
                Status::InternalServerError,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };
    let mut images_raw = Vec::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut result: Option<String> = None;
    while let Ok(Some(entry)) = dir.next_entry().await {
        let bytes = match read(entry.path()).await {
            Ok(b) => b,
            Err(e) => {
                let _ = remove_dir_all(format!("./data/{id}")).await;
                return Err(status::Custom(
                    Status::InternalServerError,
                    Json(ErrBody::new(e.to_string())),
                ));
            }
        };

        //let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_str().unwrap_or_default().to_string();
        let split: Vec<&str> = name.split('.').collect();
        if name == "output.png" {
            result = Some(general_purpose::STANDARD.encode(bytes));
            continue;
        } else if split.len() > 2 && split[split.len() - 2] == "heat_map" {
            images_raw.push((name, general_purpose::STANDARD.encode(bytes)));
        }
    }

    let result = if let Some(result) = result {
        result
    } else {
        let _ = remove_dir_all(format!("./data/{id}")).await;
        return Err(status::Custom(
            Status::InternalServerError,
            Json(ErrBody::new("Result was none".to_string())),
        ));
    };
    let mut images: Vec<String> = vec![result];

    let mut by_name: HashMap<String, String> = HashMap::new();
    for (name, img_b64) in &images_raw {
        by_name.insert(name.clone(), img_b64.clone());
    }

    let token_map_path = format!("./data/{id}/token_map.json");
    let token_map: Option<TokenMap> = match read_to_string(&token_map_path).await {
        Ok(s) => match rocket::serde::json::from_str(&s) {
            Ok(tm) => Some(tm),
            Err(e) => {
                log::warn!("Failed to parse token_map.json: {e}");
                None
            }
        },
        Err(_) => None,
    };
    if let Some(tm) = token_map {
        for m in tm.mapping {
            if let Some(img_b64) = by_name.get(&m.heatmap) {
                images.push(img_b64.clone());
                // join subword display tokens for UI; tweak joiner if you prefer
                tokens.push(if m.tokens.is_empty() {
                    m.word
                } else {
                    m.tokens.join("") // e.g., "Ġbasket" + "ball" -> "Ġbasketball"
                });
            } else {
                log::warn!("No heatmap file for {}", m.heatmap);
            }
        }
    } else {
        let _ = remove_dir_all(format!("./data/{id}")).await;
        return Err(status::Custom(
            Status::InternalServerError,
            Json(ErrBody::new("Could not find token map".to_string())),
        ));
    }
    let _ = remove_dir_all(format!("./data/{id}")).await;
    //log::info!("{}, {}", tokens.len(), images_raw.len());
    let response = GenerateResponse {
        tokens: tokens,
        images: images,
    };
    Ok((Status::Accepted, Json(response)))
}
//#[post("/generate", data = "<data>")]
//async fn generate(
//    data: Json<GenerateProps>,
//) -> Result<(Status, Json<GenerateResponse>), status::Custom<Json<ErrBody>>> {
//    let id = Uuid::new_v4();
//    match create_dir_all(format!("./data/{id}")).await {
//        Ok(()) => {}
//        Err(e) => {
//            let _ = remove_dir_all(format!("./data/{id}")).await;
//            return Err(status::Custom(
//                Status::InternalServerError,
//                Json(ErrBody::new(e.to_string())),
//            ));
//        }
//    };
//    log::info!("here");
//    let mut handle = match Command::new("./venv/Scripts/python")
//        .arg("./src-py/maps.py")
//        .arg(data.prompt.clone())
//        .arg(data.seed.clone())
//        .arg(format!("./data/{id}"))
//        .spawn()
//    {
//        Ok(h) => h,
//        Err(e) => {
//            let _ = remove_dir_all(format!("./data/{id}")).await;
//            return Err(status::Custom(
//                Status::InternalServerError,
//                Json(ErrBody::new(e.to_string())),
//            ));
//        }
//    };
//    log::info!("now here");
//    let res = match handle.wait() {
//        Ok(v) => v,
//        Err(e) => {
//            let _ = remove_dir_all(format!("./data/{id}")).await;
//            return Err(status::Custom(
//                Status::InternalServerError,
//                Json(ErrBody::new(e.to_string())),
//            ));
//        }
//    };
//    log::info!("Status Code: {res}");
//
//    let final_image = {
//        let mut dir = match read_dir(format!("./data/{id}")).await {
//            Ok(d) => d,
//            Err(e) => {
//                let _ = remove_dir_all(format!("./data/{id}")).await;
//                return Err(status::Custom(
//                    Status::InternalServerError,
//                    Json(ErrBody::new(e.to_string())),
//                ));
//            }
//        };
//        loop {
//            if let Ok(Some(entry)) = dir.next_entry().await {
//                let filetype = match entry.file_type().await {
//                    Ok(t) => t,
//                    Err(e) => {
//                        let _ = remove_dir_all(format!("./data/{id}")).await;
//                        return Err(status::Custom(
//                            Status::InternalServerError,
//                            Json(ErrBody::new(e.to_string())),
//                        ));
//                    }
//                };
//                if filetype.is_file() {
//                    let bytes = match read(entry.path()).await {
//                        Ok(b) => b,
//                        Err(e) => {
//                            let _ = remove_dir_all(format!("./data/{id}")).await;
//                            return Err(status::Custom(
//                                Status::InternalServerError,
//                                Json(ErrBody::new(e.to_string())),
//                            ));
//                        }
//                    };
//                    break general_purpose::STANDARD.encode(bytes);
//                }
//            } else {
//                return Err(status::Custom(
//                    Status::InternalServerError,
//                    Json(ErrBody::new("Couldnt find final image".to_string())),
//                ));
//            }
//        }
//    };
//    let mut dir = match read_dir(format!("./data/{id}/heatmaps")).await {
//        Ok(d) => d,
//        Err(e) => {
//            let _ = remove_dir_all(format!("./data/{id}")).await;
//            return Err(status::Custom(
//                Status::InternalServerError,
//                Json(ErrBody::new(e.to_string())),
//            ));
//        }
//    };
//    let mut images = Vec::new();
//    let mut tokens = Vec::new();
//    let mut tokens_raw = Vec::new();
//    images.push(final_image);
//    while let Ok(Some(entry)) = dir.next_entry().await {
//        let bytes = match read(entry.path()).await {
//            Ok(b) => b,
//            Err(e) => {
//                let _ = remove_dir_all(format!("./data/{id}")).await;
//                return Err(status::Custom(
//                    Status::InternalServerError,
//                    Json(ErrBody::new(e.to_string())),
//                ));
//            }
//        };
//        let file_name = entry.file_name();
//        let name = file_name.to_str().unwrap_or_default();
//        if name.starts_with("token") {
//            // Parse the filename: token-index-token_value.png
//            let parts: Vec<&str> = name.split('-').collect();
//            if parts.len() >= 3 {
//                if let Ok(index) = parts[1].parse::<u32>() {
//                    let token_value = parts[2..].join("-").replace(".png", "");
//                    tokens_raw.push((index, token_value));
//                } else {
//                    log::error!("Could not parse index");
//                }
//            } else {
//                log::error!("Parsed len is less than 3 items");
//            }
//        }
//        images.push(general_purpose::STANDARD.encode(bytes));
//    }
//    let _ = remove_dir_all(format!("./data/{id}")).await;
//    tokens_raw.sort_by(|a, b| a.0.cmp(&b.0));
//    for (_, token) in tokens_raw {
//        log::info!("{token}");
//        tokens.push(token);
//    }
//    log::info!("{}, {}", tokens.len(), images.len());
//    let response = GenerateResponse {
//        tokens: tokens,
//        images: images,
//    };
//    Ok((Status::Accepted, Json(response)))
//}

#[derive(Deserialize, Clone)]
#[serde(crate = "rocket::serde")]
struct PhysicalAttributes {
    race: Option<String>,
    clothing: Option<String>,
    age: Option<i32>,
}
impl Default for PhysicalAttributes {
    fn default() -> Self {
        Self {
            race: Some(String::new()),
            clothing: Some(String::new()),
            age: Some(0),
        }
    }
}

fn generate_prompt_from_options(prompt: &String, options: &Options) -> String {
    let mut out = String::from(prompt);
    let mut vec: Vec<String> = Vec::new();
    push_vec(&mut vec, &options.medium);
    push_vec(&mut vec, &options.genre);
    push_vec(&mut vec, &options.mood);
    if let Some(physical_attributes) = &options.physical_attributes {
        if let Some(age) = physical_attributes.age {
            vec.push(format!("{age} years old"));
        }
        if let Some(race) = &physical_attributes.race {
            vec.push(race.clone());
        }
        if let Some(clothing) = &physical_attributes.clothing {
            vec.push(clothing.clone());
        }
    }
    push_vec(&mut vec, &options.technique);
    push_vec(&mut vec, &options.lighting);
    push_vec(&mut vec, &options.resolution);
    push_vec(&mut vec, &options.setting);
    push_vec(&mut vec, &options.angle);
    if !vec.is_empty() {
        out += " -- ";
        out += vec.join(", ").as_str();
    }
    out
}

fn push_vec(out: &mut Vec<String>, vec: &Option<Vec<String>>) {
    if let Some(options) = vec {
        let mut options = options.clone();
        options.sort();
        for option in options {
            out.push(option.clone());
        }
    }
}

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct Options {
    medium: Option<Vec<String>>,
    genre: Option<Vec<String>>,
    physical_attributes: Option<PhysicalAttributes>,
    mood: Option<Vec<String>>,
    technique: Option<Vec<String>>,
    lighting: Option<Vec<String>>,
    resolution: Option<Vec<String>>,
    setting: Option<Vec<String>>,
    angle: Option<Vec<String>>,
}
impl Display for Options {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut first = true;
        writeln!(f, "{{")?;

        // helper macro – avoids repeating the same if let Some(...) pattern
        macro_rules! write_opt_vec {
            ($field:expr, $label:literal) => {
                if let Some(values) = &$field {
                    if !values.is_empty() {
                        if !first {
                            writeln!(f)?;
                        }
                        writeln!(f, "  \t\t{}: [{}]", $label, values.join(", "))?;
                        first = false;
                    }
                }
            };
        }

        write_opt_vec!(self.medium, "medium");
        write_opt_vec!(self.genre, "genre");
        write_opt_vec!(self.mood, "mood");
        write_opt_vec!(self.technique, "technique");
        write_opt_vec!(self.lighting, "lighting");
        write_opt_vec!(self.resolution, "resolution");
        write_opt_vec!(self.setting, "setting");
        write_opt_vec!(self.angle, "angle");

        // physical attributes need a little extra handling
        if let Some(pa) = &self.physical_attributes {
            if pa.race.is_some() || pa.clothing.is_some() || pa.age.is_some() {
                writeln!(f, "\t\tphysical_attributes: {{",)?;
                if let Some(race) = &pa.race {
                    writeln!(f, "\t\t\trace: \"{race}\", ")?;
                }
                if let Some(clothing) = &pa.clothing {
                    writeln!(f, "\t\t\tclothing: \"{clothing}\", ")?;
                }
                if let Some(age) = &pa.age {
                    writeln!(f, "\t\t\tage: \"{age}\", ")?;
                }
                writeln!(f, "\t\t}}",)?;
            }
        }

        writeln!(f, "\t}}")
    }
}

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct Generation {
    userid: String,
    seed: String,
    prompt: String,
    options: Options,
}

#[derive(Serialize)]
#[serde(crate = "rocket::serde")]
struct ErrBody {
    error: String,
}
impl ErrBody {
    pub fn new(error: String) -> Self {
        Self { error }
    }
}

// NOTE: needs
// * userid
// * prompt
// * options
// * seed
#[post("/create-prompt", data = "<data>")]
async fn create_prompt(
    data: Json<Generation>,
) -> Result<(Status, Json<String>), status::Custom<Json<ErrBody>>> {
    let prompt = generate_prompt_from_options(&data.prompt, &data.options);
    log::info!("{prompt}");
    Ok((Status::Accepted, Json(prompt)))
}
// NOTE: needs
// * userid
// * prompt
// * options
// * seed
#[post("/create-gpt-prompt", data = "<data>")]
async fn create_gpt_prompt(
    data: Json<Generation>,
) -> Result<(Status, Json<String>), status::Custom<Json<ErrBody>>> {
    let client = Client::new();
    // single
    let prompt = format!(
        r#"
        You are an expert prompt engineer for text‑to‑image models.

        **Task**  
        Add to the ORIGINAL_PROMPT so that it is crystal‑clear, compact (≤75 words),
        and contains every OPTION the user supplied without altering the ORIGINAL_PROMPT.  
        • Keep the imagery, subject, and intent of the ORIGINAL_PROMPT intact.  
        • Insert the OPTION values at sensible places; if an option is missing, skip it.  
        • Do **not** invent new details.  
        • Merge duplicate words and strip superfluous adjectives.  
        • Use commas to separate major phrases; avoid lists in brackets.  
        • Finish with a period and output **exactly one line**—no JSON, no numbering,
        no commentary, no line breaks.

        ---

        ORIGINAL_PROMPT: {}

        OPTIONS  
        {}

        ---  
        "#,
        data.prompt, data.options
    );
    log::info!("{prompt}");
    let request = match CreateCompletionRequestArgs::default()
        .model("gpt-4o-mini-2024-07-18")
        .prompt(prompt)
        .max_tokens(60_u32)
        .build()
    {
        Ok(v) => v,
        Err(e) => {
            return Err(status::Custom(
                Status::BadRequest,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };

    let response = match client.completions().create(request).await {
        Ok(v) => v,
        Err(e) => {
            return Err(status::Custom(
                Status::BadRequest,
                Json(ErrBody::new(e.to_string())),
            ));
        }
    };

    println!("\nResponse (single):\n");
    for choice in &response.choices {
        println!("{}", choice.text);
    }
    Ok((Status::Accepted, Json(response.choices[0].clone().text)))
}

#[launch]
fn rocket() -> _ {
    dotenv().ok();
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();
    log::info!("----- START -----");
    rocket::build().mount("/", routes![generate, create_prompt, create_gpt_prompt])
}
