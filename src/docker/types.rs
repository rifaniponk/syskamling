//! Types returned by the Docker Engine API.
//!
//! Field names match the daemon JSON exactly; `rename_all = "camelCase"`
//! handles the Engine's usual convention.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ContainerSummary {
    pub id: String,
    pub names: Vec<String>,
    pub image: String,
    #[serde(default)]
    pub image_id: String,
    pub state: String,
    pub status: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub ports: Vec<Port>,
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Port {
    #[serde(default)]
    pub ip: Option<String>,
    #[serde(default)]
    pub private_port: Option<u16>,
    #[serde(default)]
    pub public_port: Option<u16>,
    #[serde(default)]
    pub typ: Option<String>,
}

impl ContainerSummary {
    /// The daemon prefixes names with `/`; strip it and take the first name.
    pub fn short_name(&self) -> String {
        self.names
            .first()
            .map(|n| n.trim_start_matches('/').to_string())
            .unwrap_or_else(|| self.id[..12].to_string())
    }

    pub fn short_id(&self) -> &str {
        &self.id[..12.min(self.id.len())]
    }

    /// Human-readable port mapping, e.g. `0.0.0.0:8080->80/tcp`.
    pub fn port_summary(&self) -> String {
        self.ports
            .iter()
            .filter_map(|p| {
                let public = p.public_port?;
                let private = p.private_port?;
                let proto = p.typ.as_deref().unwrap_or("tcp");
                Some(format!("{public}->{private}/{proto}"))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSummary {
    pub id: String,
    #[serde(default)]
    pub repo_tags: Vec<String>,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub size: i64,
}

impl ImageSummary {
    pub fn tag(&self) -> String {
        self.repo_tags
            .first()
            .cloned()
            .unwrap_or_else(|| self.id[..19.min(self.id.len())].to_string())
    }

    pub fn short_id(&self) -> &str {
        let id = self.id.trim_start_matches("sha256:");
        &id[..12.min(id.len())]
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VolumeEntry {
    pub name: String,
    pub driver: String,
    #[serde(default)]
    pub created: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemDf {
    #[serde(default)]
    pub layers_size: i64,
    #[serde(default)]
    pub containers_summary: Vec<DfContainer>,
    #[serde(default)]
    pub images: Vec<DfImage>,
    #[serde(default)]
    pub volumes_summary: DfVolumes,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DfContainer {
    pub state: String,
    pub size: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DfImage {
    #[serde(default)]
    pub size: i64,
    #[serde(default)]
    pub shared_size: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DfVolumes {
    #[serde(default)]
    pub size: i64,
}
