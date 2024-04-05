# Enable apis
resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "cloud_scheduler" {
  service = "cloudscheduler.googleapis.com"
}

# Create resources
resource "google_cloud_run_v2_service" "dd-service" {
  name     = "dd-service"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west1"

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/dynamicdealmakers-7012254/dd-repo/dd_webshop:v0.0.14"

      env {
        name  = "TF_VAR_api_key"
        value = var.api_key
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

resource "google_cloud_run_service_iam_member" "scheduler_job_runner" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_run_service_iam_member" "wesley" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "user:wboelrijk@xccelerated.io"
}

# whitelist all
resource "google_cloud_run_service_iam_member" "all" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "allUsers"
}

resource "google_cloud_scheduler_job" "dd-scheduler" {
  name     = "dd-scheduler"
  schedule = "*/1 * * * *"
  time_zone = "Europe/Amsterdam"
  region = "europe-west1"

  http_target {
    http_method = "GET"
    uri = "${google_cloud_run_v2_service.dd-service.uri}/scrape-data"
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = google_cloud_run_v2_service.dd-service.uri  
    }
  }

  depends_on = [
    google_cloud_run_v2_service.dd-service,
  ]
}
