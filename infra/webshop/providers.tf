terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.19.0"
    }
  }

  backend "gcs" {
    bucket = "dd_tf"
    prefix = "webshop/terraform/state"
  }
}

provider "google" {
  project = "dynamicdealmakers-7012254"
  region  = "europe-west1"
}