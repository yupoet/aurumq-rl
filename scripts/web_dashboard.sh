#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../web"
npm install --silent
exec npm run dev
