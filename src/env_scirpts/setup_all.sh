#!/usr/bin/env bash
# scripts/setup_all.sh
# Полная настройка проекта: окружение + линтеры.
# Запуск: bash scripts/setup_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo " Шаг 1/2: окружение"
bash "$SCRIPT_DIR/setup_env.sh"

echo ""
echo " Шаг 2/2: линтеры"
bash "$SCRIPT_DIR/setup_linters.sh"

echo ""
echo "   Готово. Дальнейшие коммиты будут автоматически проверяться pre-commit."
echo "   Для ручного прогона: poetry run pre-commit run --all-files"