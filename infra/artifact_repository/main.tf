terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.19.0"
    }
  }

  backend "gcs" {
    bucket = "dd_tf"
    prefix = "artifact/terraform/state"
  }
}

provider "google" {
  project = "dynamicdealmakers-7012254"
  region  = "europe-west4"
}

resource "google_artifact_registry_repository" "dd-repo" {
  repository_id = "dd-repo"
  format        = "DOCKER"
}