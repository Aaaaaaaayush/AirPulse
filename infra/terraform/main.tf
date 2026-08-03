terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# S3 Bucket for Raw Data Lake
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "raw" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Project     = "AirPulse"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Block all public access to the bucket (security best practice)
resource "aws_s3_bucket_public_access_block" "raw_public_block" {
  bucket = aws_s3_bucket.raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule: expire raw JSON objects after 90 days to keep costs minimal
resource "aws_s3_bucket_lifecycle_configuration" "raw_lifecycle" {
  bucket = aws_s3_bucket.raw.id

  rule {
    id     = "expire-raw-after-90-days"
    status = "Enabled"

    filter {}

    expiration {
      days = 90
    }
  }
}

# ---------------------------------------------------------------------------
# IAM User & Least-Privilege Policy for Ingestion Pipeline
# ---------------------------------------------------------------------------
resource "aws_iam_user" "ingestion_user" {
  name = "airpulse-ingestion-${var.environment}"
  tags = {
    Project     = "AirPulse"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_access_key" "ingestion_key" {
  user = aws_iam_user.ingestion_user.name
}

resource "aws_iam_user_policy" "ingestion_policy" {
  name = "airpulse-ingestion-s3-access"
  user = aws_iam_user.ingestion_user.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.raw.arn,
          "${aws_s3_bucket.raw.arn}/*"
        ]
      }
    ]
  })
}
