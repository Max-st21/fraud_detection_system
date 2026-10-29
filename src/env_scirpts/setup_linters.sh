#!/usr/bin/env bash
# scripts/setup_linters.sh
# Настройка линтеров и pre-commit: black, isort, flake8.
# Запуск: bash scripts/setup_linters.sh

set -euo pipefail

echo "=================================================="
echo " 1. Установка линтеров через Poetry"
echo "=================================================="
# Добавляем в dev-зависимости (идемпотентно: повторный add не сломает lock)
poetry add --group dev black isort flake8 pre-commit 2>/dev/null || \
    poetry add --dev black isort flake8 pre-commit
echo "  Линтеры установлены"

echo ""
echo "=================================================="
echo " 2. Генерация/проверка .pre-commit-config.yaml"
echo "=================================================="
CONFIG_FILE=".pre-commit-config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Файл $CONFIG_FILE не найден — создаю..."
    cat > "$CONFIG_FILE" <<'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
EOF
    echo "  Создан $CONFIG_FILE"
else
    echo "ℹ  $CONFIG_FILE уже существует — пропускаю создание"
fi

echo ""
echo "=================================================="
echo " 3. Прогон black и isort по проекту"
echo "=================================================="
# Запускаем внутри Poetry-окружения, чтобы не зависеть от глобальных пакетов
poetry run black .
poetry run isort .
echo "  Код отформатирован"

echo ""
echo "=================================================="
echo " 4. Установка pre-commit хука"
echo "=================================================="
poetry run pre-commit install
echo "  Хук установлен в .git/hooks/pre-commit"

echo ""
echo "=================================================="
echo " 5. Первый прогон pre-commit по всем файлам"
echo "=================================================="
# --all-files: проверяем весь проект, а не только staged
poetry run pre-commit run --all-files || {
    echo ""
    echo "   pre-commit нашёл замечания."
    echo "   Многие из них исправляются автоматически (trailing-whitespace, end-of-file-fixer)."
    echo "   Повторный запуск обычно проходит чисто:"
    echo "   poetry run pre-commit run --all-files"
}

echo ""
echo "  setup_linters.sh завершён"