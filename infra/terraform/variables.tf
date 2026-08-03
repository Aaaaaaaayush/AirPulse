variable "aws_region" {
  description = "AWS region for S3 and IAM resources"
  type        = string
  default     = "ap-south-1"
}

variable "bucket_name" {
  description = "Unique name of the S3 raw data bucket"
  type        = string
  default     = "airpulse-raw"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}
