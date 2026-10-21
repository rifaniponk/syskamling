//! Types returned by the Docker Engine API.
//!
//! Field names match the daemon JSON exactly; `rename_all = "camelCase"`
//! handles the Engine's usual convention.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerSummary {
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(rename = "Names")]
    pub names: Vec<String>,
    #[serde(rename = "Image")]
    pub image: String,
    #[serde(rename = "State")]
    pub state: String,
    #[serde(rename = "Status")]
    pub status: String,
    #[serde(default, rename = "Ports")]
    pub ports: Vec<Port>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Port {
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
#[serde(rename_all = "camelCase")]
pub struct ImageSummary {
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(default, rename = "RepoTags")]
    pub repo_tags: Vec<String>,
    #[serde(rename = "Created")]
    pub created: i64,
    #[serde(rename = "Size")]
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
#[serde(rename_all = "camelCase")]
pub struct VolumeEntry {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Driver")]
    pub driver: String,
    #[serde(default, rename = "CreatedAt")]
    pub created: String,
}
