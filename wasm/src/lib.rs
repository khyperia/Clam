#[cfg(target_arch = "wasm32")]
use clam5::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn start() -> Result<(), JsValue> {
    console_log::init().unwrap();
    console_error_panic_hook::set_once();
    run().await.map_err(|e| JsError::new(&e.to_string()))?;
    Ok(())
}
