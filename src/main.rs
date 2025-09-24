use std::{
    collections::{HashMap, VecDeque},
    fmt::{self, Display, Formatter},
    fs::FileType,
    io::Error,
    process::Command,
    thread,
    time::Duration,
};

use async_openai::{
    Client,
    types::{CreateCompletionRequestArgs, CreateImageRequestArgs, ImageResponseFormat, ImageSize},
};
use base64::{Engine, engine::general_purpose};
use dotenv::dotenv;
use rocket::{
    State,
    fairing::AdHoc,
    futures::{SinkExt, StreamExt, executor::block_on},
    http::{Status, ext::IntoCollection},
    response::status,
    serde::{
        self, Deserialize, Serialize,
        json::{self, Json, serde_json::json},
    },
    tokio::{
        self,
        fs::{create_dir_all, read, read_dir, read_to_string, remove_dir_all},
        sync::{
            Mutex, RwLock,
            mpsc::{Sender, channel},
        },
        time::interval,
    },
};
use std::sync::Arc;
use uuid::Uuid;
use ws::Message;

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

enum GenerateErr {
    Generic { msg: String },
}

async fn generate(data: &Generate) -> Result<GenerateResponse, GenerateErr> {
    let id = Uuid::new_v4();
    match create_dir_all(format!("./data/{id}")).await {
        Ok(()) => {}
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(GenerateErr::Generic { msg: e.to_string() });
            //return Err(status::Custom(
            //    Status::InternalServerError,
            //    Json(ErrBody::new(e.to_string())),
            //));
        }
    };
    log::info!("here");
    let mut handle = match Command::new("./venv/bin/python")
        .arg("./src-py/maps-test.py")
        .arg(data.prompt.clone())
        .arg(data.seed.clone())
        .arg(format!("./data/{id}"))
        .spawn()
    {
        Ok(h) => h,
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(GenerateErr::Generic { msg: e.to_string() });
            //return Err(status::Custom(
            //    Status::InternalServerError,
            //    Json(ErrBody::new(e.to_string())),
            //));
        }
    };
    log::info!("now here");
    let res = tokio::task::spawn_blocking(move || handle.wait()).await;
    let res = match res {
        Ok(v) => match v {
            Ok(v) => v,
            Err(e) => {
                let _ = remove_dir_all(format!("./data/{id}")).await;
                return Err(GenerateErr::Generic { msg: e.to_string() });
            }
        },
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(GenerateErr::Generic { msg: e.to_string() });
        }
    };

    //return Err(status::Custom(
    //    Status::InternalServerError,
    //    Json(ErrBody::new(e.to_string())),
    //));

    log::info!("Status Code: {res}");

    let mut dir = match read_dir(format!("./data/{id}")).await {
        Ok(d) => d,
        Err(e) => {
            let _ = remove_dir_all(format!("./data/{id}")).await;
            return Err(GenerateErr::Generic { msg: e.to_string() });
            //return Err(status::Custom(
            //    Status::InternalServerError,
            //    Json(ErrBody::new(e.to_string())),
            //));
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
                return Err(GenerateErr::Generic { msg: e.to_string() });
                //return Err(status::Custom(
                //    Status::InternalServerError,
                //    Json(ErrBody::new(e.to_string())),
                //));
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
        return Err(GenerateErr::Generic {
            msg: "Result was none".to_string(),
        });
        //return Err(status::Custom(
        //    Status::InternalServerError,
        //    Json(ErrBody::new("Result was none".to_string())),
        //));
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
        return Err(GenerateErr::Generic {
            msg: "Could not find token map".to_string(),
        });
        //return Err(status::Custom(
        //    Status::InternalServerError,
        //    Json(ErrBody::new("Could not find token map".to_string())),
        //));
    }
    let _ = remove_dir_all(format!("./data/{id}")).await;
    //log::info!("{}, {}", tokens.len(), images_raw.len());
    let response = GenerateResponse {
        tokens: tokens,
        images: images,
    };
    //Ok((Status::Accepted, Json(response)))
    Ok(response)
}

// NOTE: needs
// * userid
// * prompt
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

#[derive(Deserialize, Clone, Debug, Serialize)]
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

#[derive(Deserialize, Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", crate = "rocket::serde")]
struct Generate {
    userid: String,
    comparison_id: String,
    generation_id: String,
    prompt: String,
    seed: String,
    options: Options,
}

struct AppState {
    job_queue: Arc<RwLock<VecDeque<WSJob<Generate>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", crate = "rocket::serde")]
enum ClientMsg {
    Generate {
        prompt: String,
        comparison_id: String,
        generation_id: String,
        userid: String,
        seed: String,
        options: Options,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", crate = "rocket::serde")]
enum ServerMsg {
    GenerationComplete,
}

struct WSJob<T> {
    job_id: Uuid,
    data: T,
    transmitter: Sender<ServerMsg>,
}

#[get("/ws-connect")]
async fn ws_connect(ws: ws::WebSocket, state: &State<AppState>) -> ws::Channel<'static> {
    let job_queue = state.job_queue.clone();
    ws.channel(move |mut stream| {
        Box::pin(async move {
            let (mut sink, mut source) = stream.split();
            let (tx, mut rx) = channel::<ServerMsg>(128);
            tokio::spawn(async move {
                while let Some(message) = rx.recv().await {
                        let err = sink
                            .send(Message::Text(json::to_string(&message).unwrap()))
                            .await;
                        if let Err(err) = err {
                            log::error!("{err}");
                        }
                }
                rx.close();
            });
            while let Some(frame) = source.next().await {
                match frame {
                    Ok(Message::Text(ref s)) => match json::from_str(s.as_str()) {
                        Ok(ClientMsg::Generate {
                            prompt,
                            comparison_id,
                            generation_id,
                            userid,
                            seed,
                            options,
                        }) => {
                            log::info!("prompt={}", prompt);
                            {
                                let mut job_queue_guard = job_queue.write().await;
                                job_queue_guard.push_back(WSJob {
                                    job_id: Uuid::new_v4(),
                                    data: Generate {
                                        prompt,
                                        comparison_id,
                                        generation_id,
                                        userid,
                                        seed,
                                        options,
                                    },
                                    transmitter: tx.clone(),
                                });
                            }
                        }
                        _ => {
                            log::error!("Unhandled message type: {}", frame.unwrap());
                        }
                    },
                    _ => {}
                }
            }
            Ok(())
        })
    })
}

async fn upload_png_b64(
    s3: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    b64: &str,
) -> Result<(), aws_sdk_s3::Error> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("bad base64");
    s3.put_object()
        .bucket(bucket)
        .key(key)
        .body(bytes.into())
        .content_type("image/png")
        .send()
        .await?;
    Ok(())
}

#[launch]
async fn rocket() -> _ {
    dotenv().ok();
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    log::info!("----- START -----");
    let job_queue = Arc::new(RwLock::new(VecDeque::<WSJob<Generate>>::new()));
    let job_queue_clone = job_queue.clone();
    rocket::build()
        .manage(AppState { job_queue })
        .attach(AdHoc::on_ignite("queue-worker", |rocket| async {
            tokio::spawn(async move {
                let db = sqlx::postgres::PgPoolOptions::new()
                    .max_connections(10)
                    .connect(&std::env::var("DATABASE_URL").expect("DATABASE_URL"))
                    .await
                    .expect("db connect");
                let config = aws_config::load_from_env().await;
                let s3 = aws_sdk_s3::Client::new(&config);
                let bucket = std::env::var("BUCKET").expect("BUCKET");
                let mut tick = interval(Duration::from_secs(1));
                loop {
                    tick.tick().await;
                    let front = {
                        let mut queue = job_queue_clone.write().await;
                        queue.pop_front()
                    };
                    if let Some(front) = front {
                        log::info!("FOUND FRONT: {}, {:?}", front.job_id, front.data);
                        match generate(&front.data).await {
                            Ok(res) => {
                                log::info!("FINISHED GENERATING");
                                //let data: { tokens: string[], images: string[] } = await result2.json();

                                //console.log(data.images.length)
                                //console.log(data.tokens.length)
                                //console.log(comparison_id)
                                let base: String = format!(
                                    "data/{}/{}",
                                    front.data.comparison_id, front.data.generation_id
                                );
                                let _ = upload_png_b64(
                                    &s3,
                                    &bucket,
                                    &format!("{base}/output.png"),
                                    &res.images[0],
                                )
                                .await;
                                let mut images: Vec<String> = Vec::with_capacity(res.images.len());

                                for (idx, img_b64) in res.images.iter().enumerate().skip(1) {
                                    let key = format!("{base}/{}.png", idx - 1);
                                    let _ = upload_png_b64(&s3, &bucket, &key, &img_b64).await;
                                    images.push(key);
                                }
                                let res = sqlx::query!(
                                    r#"
                                    UPDATE generation
                                    SET output=$1, prompt=$2, options=$3, images=$4, tokens=$5
                                    WHERE id=$6
                                "#,
                                    format!("{base}/output.png"),
                                    front.data.prompt,
                                    json!(front.data.options),
                                    &images,
                                    &res.tokens,
                                    Uuid::parse_str(&front.data.generation_id).expect("ID was not a UUID")
                                );
                                if let Err(e) = res.execute(&db).await {
                                    log::error!("{e}");
                                }
                                let err =
                                    front.transmitter.send(ServerMsg::GenerationComplete).await;
                                if let Err(err) = err {
                                    log::error!("Error sending event: {err}");
                                }
                            }
                            Err(e) => match e {
                                GenerateErr::Generic { msg } => {
                                    log::error!("Error sending event: {msg}");
                                }
                            },
                        }
                    }
                }
            });
            rocket
        }))
        .mount("/", routes![create_prompt, create_gpt_prompt, ws_connect])
}
