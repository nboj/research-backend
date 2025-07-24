use std::{
    fmt::{self, Display, Formatter},
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
    http::Status,
    response::status,
    serde::{Deserialize, Serialize, json::Json},
    tokio::fs::{create_dir_all, read, read_dir, remove_dir_all},
};
use uuid::Uuid;

#[macro_use]
extern crate rocket;

#[derive(Deserialize)]
#[serde(crate = "rocket::serde")]
struct GenerateProps {
    userid: String,
    seed: String,
    prompt: String,
}

// NOTE: needs
// * userid
// * prompt
#[post("/generate", data = "<data>")]
async fn generate(
    data: Json<GenerateProps>,
) -> Result<(Status, Json<Vec<String>>), status::Custom<Json<ErrBody>>> {
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
    let mut handle = match Command::new("./venv/bin/python")
        .arg("./src-py/main.py")
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

    let mut images_b64 = Vec::new();
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
        images_b64.push(general_purpose::STANDARD.encode(bytes));
    }
    let _ = remove_dir_all(format!("./data/{id}")).await;
    Ok((Status::Accepted, Json(images_b64)))
}

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
    log::info!("");
    log::info!("Prompt: {}", data.prompt);
    log::info!("Seed: {}", data.seed);
    log::info!("UserID: {}", data.userid);
    log::info!(
        "opions.Medium: {}",
        data.options.medium.clone().unwrap_or_default().join(", ")
    );
    log::info!(
        "opions.Genre: {}",
        data.options.genre.clone().unwrap_or_default().join(", ")
    );
    log::info!(
        "opions.PhysicalAttributes: {}",
        data.options
            .physical_attributes
            .clone()
            .unwrap_or_default()
            .age
            .unwrap_or_default()
    );
    log::info!(
        "opions.PhysicalAttributes: {}",
        data.options
            .physical_attributes
            .clone()
            .unwrap_or_default()
            .race
            .unwrap_or_default()
    );
    log::info!(
        "opions.PhysicalAttributes: {}",
        data.options
            .physical_attributes
            .clone()
            .unwrap_or_default()
            .clothing
            .unwrap_or_default()
    );
    log::info!(
        "opions.Mood: {}",
        data.options.mood.clone().unwrap_or_default().join(", ")
    );
    log::info!(
        "opions.Technique: {}",
        data.options
            .technique
            .clone()
            .unwrap_or_default()
            .join(", ")
    );
    log::info!(
        "opions.Lighting: {}",
        data.options.lighting.clone().unwrap_or_default().join(", ")
    );
    log::info!(
        "opions.Resolution: {}",
        data.options
            .resolution
            .clone()
            .unwrap_or_default()
            .join(", ")
    );
    log::info!("");
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
        FINAL_REWRITTEN_PROMPT:
        "#,
        data.prompt, data.options
    );
    log::info!("{prompt}");
    let request = match CreateCompletionRequestArgs::default()
        .model("gpt-4o-mini-2024-07-18")
        .prompt(prompt)
        .max_tokens(40_u32)
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
    rocket::build().mount("/", routes![generate, create_prompt])
}
