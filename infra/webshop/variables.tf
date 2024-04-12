# variable "database_url" {
#   type      = string
#   sensitive = true
# }

variable "api_key" {
  type      = string
  sensitive = true
}

variable "postgres_conn_string" {
  type      = string
  sensitive = true
}

variable "secret_name_postgres" {
  type      = string
  sensitive = true
}

variable "secret_name_sa" {
  type      = string
  sensitive = true
}

variable "pg_host" {
  type      = string
  sensitive = true
}
variable "pg_user" {
  type      = string
  sensitive = true
}
variable "pg_pass" {
  type      = string
  sensitive = true
}
