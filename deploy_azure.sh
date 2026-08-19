#!/usr/bin/env bash
set -e

# Configuration Variables
RESOURCE_GROUP="depi_demo"
LOCATION="germanywestcentral"
RAND=$RANDOM
ACR_NAME="acrrag$RAND"
STORAGE_ACCOUNT="stgchroma$RAND"
FILE_SHARE_NAME="chroma-share"
KEY_VAULT_NAME="kvrag$RAND"
ENVIRONMENT_NAME="cae-rag-env"
APP_NAME="rag-backend-api"

# Load secrets from local .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-your_azure_openai_api_key_here}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://ah30309142502238-8748-resource.openai.azure.com/openai/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-o4-mini}"

echo "=== 1. Creating Resource Group ==="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "=== 2. Creating Azure Container Registry (ACR) ==="
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true

echo "=== 3. Creating Key Vault and storing secrets ==="
az keyvault create --resource-group "$RESOURCE_GROUP" --name "$KEY_VAULT_NAME" --location "$LOCATION"
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "OpenAiApiKey" --value "$OPENAI_API_KEY"

echo "=== 4. Creating Storage Account & Azure Files Share for Chroma DB ==="
az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2

STORAGE_KEY=$(az storage account keys list --resource-group "$RESOURCE_GROUP" --account-name "$STORAGE_ACCOUNT" --query "[0].value" --output tsv)

az storage share create \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --name "$FILE_SHARE_NAME"

echo "=== 5. Creating Container Apps Environment ==="
az containerapp env create \
    --name "$ENVIRONMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"

echo "=== 6. Linking Azure Files Share to Container Apps Environment ==="
az containerapp env storage set \
    --name "$ENVIRONMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --storage-name chroma-mount \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$FILE_SHARE_NAME" \
    --access-mode ReadWrite

echo "=== 7. Building & Pushing Docker Image to ACR ==="
az acr build --registry "$ACR_NAME" --image "rag-backend:v1" .

echo "=== 8. Deploying Azure Container App ==="
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENVIRONMENT_NAME" \
    --image "$ACR_NAME.azurecr.io/rag-backend:v1" \
    --registry-server "$ACR_NAME.azurecr.io" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 2 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --env-vars \
        OPENAI_API_KEY="secretref:openai-api-key" \
        OPENAI_BASE_URL="$OPENAI_BASE_URL" \
        OPENAI_MODEL="$OPENAI_MODEL" \
        CHROMA_PERSIST_DIR="/data/chroma_db" \
    --secrets \
        openai-api-key="keyvaultref:$KEY_VAULT_NAME.vault.azure.net/secrets/OpenAiApiKey" \
    --system-assigned

echo "=== 9. Mounting Persistent Chroma DB Volume ==="
az containerapp volume mount set \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --volume-name chroma-volume \
    --storage-name chroma-mount \
    --mount-path /data/chroma_db

echo "=== Deployment Completed Successfully! ==="
FQDN=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" --output tsv)
echo "Public API Base URL: https://$FQDN"
echo "OpenAPI Documentation: https://$FQDN/docs"
