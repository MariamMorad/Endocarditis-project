# Azure Deployment Script for PowerShell
$ErrorActionPreference = "Stop"

$RESOURCE_GROUP = "depi_demo"
$LOCATION = "germanywestcentral"
$RAND = Get-Random -Minimum 1000 -Maximum 9999
$ACR_NAME = "acrrag$RAND"
$STORAGE_ACCOUNT = "stgchroma$RAND"
$FILE_SHARE_NAME = "chroma-share"
$KEY_VAULT_NAME = "kvrag$RAND"
$ENVIRONMENT_NAME = "cae-rag-env"
$APP_NAME = "rag-backend-api"

# Load variables from .env if present
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
}

$OPENAI_API_KEY = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "your_azure_openai_api_key_here" }
$OPENAI_BASE_URL = if ($env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL } else { "https://ah30309142502238-8748-resource.openai.azure.com/openai/v1" }
$OPENAI_MODEL = if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "o4-mini" }

Write-Host "=== 1. Creating Resource Group ==="
az group create --name $RESOURCE_GROUP --location $LOCATION

Write-Host "=== 2. Creating Azure Container Registry (ACR) ==="
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

Write-Host "=== 3. Creating Key Vault and storing secrets ==="
az keyvault create --resource-group $RESOURCE_GROUP --name $KEY_VAULT_NAME --location $LOCATION
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "OpenAiApiKey" --value $OPENAI_API_KEY

Write-Host "=== 4. Creating Storage Account & Azure Files Share for Chroma DB ==="
az storage account create `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION `
    --sku Standard_LRS `
    --kind StorageV2

$STORAGE_KEY = az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT --query "[0].value" --output tsv

az storage share create `
    --account-name $STORAGE_ACCOUNT `
    --account-key $STORAGE_KEY `
    --name $FILE_SHARE_NAME

Write-Host "=== 5. Creating Container Apps Environment ==="
az containerapp env create `
    --name $ENVIRONMENT_NAME `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION

Write-Host "=== 6. Linking Azure Files Share to Container Apps Environment ==="
az containerapp env storage set `
    --name $ENVIRONMENT_NAME `
    --resource-group $RESOURCE_GROUP `
    --storage-name chroma-mount `
    --azure-file-account-name $STORAGE_ACCOUNT `
    --azure-file-account-key $STORAGE_KEY `
    --azure-file-share-name $FILE_SHARE_NAME `
    --access-mode ReadWrite

Write-Host "=== 7. Building & Pushing Docker Image to ACR ==="
az acr build --registry $ACR_NAME --image "rag-backend:v1" .

Write-Host "=== 8. Deploying Azure Container App ==="
az containerapp create `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --environment $ENVIRONMENT_NAME `
    --image "$ACR_NAME.azurecr.io/rag-backend:v1" `
    --registry-server "$ACR_NAME.azurecr.io" `
    --target-port 8000 `
    --ingress external `
    --min-replicas 1 `
    --max-replicas 2 `
    --cpu 1.0 `
    --memory 2.0Gi `
    --env-vars `
        OPENAI_API_KEY="secretref:openai-api-key" `
        OPENAI_BASE_URL="$OPENAI_BASE_URL" `
        OPENAI_MODEL="$OPENAI_MODEL" `
        CHROMA_PERSIST_DIR="/data/chroma_db" `
    --secrets `
        openai-api-key="keyvaultref:$KEY_VAULT_NAME.vault.azure.net/secrets/OpenAiApiKey" `
    --system-assigned

Write-Host "=== 9. Mounting Persistent Chroma DB Volume ==="
az containerapp volume mount set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --volume-name chroma-volume `
    --storage-name chroma-mount `
    --mount-path /data/chroma_db

Write-Host "=== Deployment Completed Successfully! ==="
$FQDN = az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" --output tsv
Write-Host "Public API Base URL: https://$FQDN"
Write-Host "OpenAPI Documentation: https://$FQDN/docs"
