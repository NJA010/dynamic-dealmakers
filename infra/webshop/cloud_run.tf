# Enable apis
resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "cloud_scheduler" {
  service = "cloudscheduler.googleapis.com"
}

# Create recourses
resource "google_cloud_run_v2_service" "dd-service" {
  name     = "dd-service"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west4"

  template {
    containers {
      image = "europe-west4-docker.pkg.dev/dynamicdealmakers-7012254/dd-repo/dd_webshop:latest"

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

resource "google_service_account" "scheduler" {
  account_id   = "scheduler"
  display_name = "Scheduler Service Account"
}

resource "google_project_iam_member" "scheduler_job_runner" {
  project = "dynamicdealmakers-7012254"
  role    = "roles/cloudscheduler.jobRunner"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "dd-scheduler" {
  name     = "dd-scheduler"
  schedule = "*/1 * * * *"
  time_zone = "Europe/Amsterdam"
  region = "us-central1"

  http_target {
    http_method = "GET"
    uri = "${google_cloud_run_v2_service.dd-service.uri}/scrape-data"
    oidc_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [
    google_cloud_run_v2_service.dd-service,
  ]
}