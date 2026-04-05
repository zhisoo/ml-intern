#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: ./create-pr.sh \"PR Title\" [\"Optional description\"]${NC}"
    echo ""
    echo "Example:"
    echo "  ./create-pr.sh \"Fix authentication bug\" \"This fixes the dev mode auth issue\""
    exit 1
fi

TITLE="$1"
DESCRIPTION="${2:-}"

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$BRANCH" = "main" ]; then
    echo -e "${RED}Error: You're on the main branch. Please create a feature branch first.${NC}"
    exit 1
fi

echo -e "${BLUE}Creating PR for branch: ${GREEN}$BRANCH${NC}"
echo -e "${BLUE}Title: ${GREEN}$TITLE${NC}"

# Get HF_TOKEN from .env
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)

if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN not found in .env${NC}"
    exit 1
fi

# Get list of changed files
echo -e "${BLUE}Detecting changed files...${NC}"
CHANGED_FILES=$(git diff --name-only main.."$BRANCH")

if [ -z "$CHANGED_FILES" ]; then
    echo -e "${RED}Error: No changes detected between main and $BRANCH${NC}"
    exit 1
fi

echo -e "${BLUE}Changed files:${NC}"
echo "$CHANGED_FILES" | while read -r file; do
    echo -e "  ${GREEN}$file${NC}"
done

# Create PR using HuggingFace API with actual file operations
echo -e "${BLUE}Creating pull request with file changes...${NC}"

PR_URL=$(HF_TOKEN="$HF_TOKEN" uv run python - <<EOF
from huggingface_hub import HfApi, CommitOperationAdd
import os
import sys

api = HfApi(token=os.environ.get('HF_TOKEN'))

# Get changed files from stdin
changed_files = """$CHANGED_FILES""".strip().split('\n')

operations = []
for file_path in changed_files:
    file_path = file_path.strip()
    if not file_path:
        continue

    try:
        with open(file_path, 'rb') as f:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=file_path,
                    path_or_fileobj=f.read()
                )
            )
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, skipping", file=sys.stderr)
        continue

if not operations:
    print("Error: No valid file operations", file=sys.stderr)
    sys.exit(1)

description = """$DESCRIPTION"""
commit_message = """$TITLE"""

# Create PR with actual file changes
try:
    result = api.create_commit(
        repo_id='smolagents/ml-agent',
        repo_type='space',
        commit_message=commit_message,
        commit_description=description if description.strip() else f"Changes from branch $BRANCH",
        operations=operations,
        create_pr=True,
    )
    print(result.pr_url)
except Exception as e:
    print(f"Error creating PR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create PR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ PR created successfully!${NC}"
echo -e "${GREEN}  $PR_URL${NC}"
