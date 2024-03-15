resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

resource "google_cloud_run_v2_service" "dd-service" {
  name     = "dd-service"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west4"

  template {
    containers {
      image = "europe-west4-docker.pkg.dev/caio-iac-training-2269012/simulator-repo/simulator:latest"

      env {
        name  = "DATABASE_URL"
        value = var.database_url
      }

    }
  }

  depends_on = [
    google_project_service.cloud_run,
  ]
}
