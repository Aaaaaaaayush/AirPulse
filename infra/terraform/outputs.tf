output "bucket_name" {
  description = "Name of the created S3 bucket"
  value       = aws_s3_bucket.raw.id
}

output "bucket_arn" {
  description = "ARN of the created S3 bucket"
  value       = aws_s3_bucket.raw.arn
}

output "ingestion_iam_user" {
  description = "IAM user name for ingestion service"
  value       = aws_iam_user.ingestion_user.name
}

output "ingestion_access_key_id" {
  description = "AWS Access Key ID for ingestion user"
  value       = aws_iam_access_key.ingestion_key.id
}

output "ingestion_secret_access_key" {
  description = "AWS Secret Access Key for ingestion user (sensitive)"
  value       = aws_iam_access_key.ingestion_key.secret
  sensitive   = true
}
