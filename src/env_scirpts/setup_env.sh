#!/usr/bin/env bash
# scripts/setup_env.sh
# Автоматизированная установка окружения проекта через Poetry.
# Запуск: bash scripts/setup_env.sh

set -euo pipefail

echo "=================================================="
echo " 1. Проверка наличия Poetry"
echo "=================================================="
if ! command -v poetry >/dev/null 2>&1; then
    echo "  Poetry не найден. Устанавливаю..."
    # Официальный установщик Poetry
    curl -sSL https://install.python-poetry.org | python3 -
    # Добавляем в PATH для текущей сессии
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v poetry >/dev/null 2>&1; then
        echo "  Не удалось установить Poetry. Установите вручную:"
        echo "   https://python-poetry.org/docs/#installation"
        exit 1
    fi
fi
echo "Poetry: $(poetry --version)"

echo ""
echo "=================================================="
echo " 2. Установка зависимостей из lock-файла"
echo "=================================================="
# --no-root: не устанавливать сам проект как пакет (как в вашем отчёте)
poetry install --no-root
echo "Зависимости установлены"

echo ""
echo "=================================================="
echo " 3. Проверка активации виртуального окружения"
echo "=================================================="
poetry env info
echo ""
echo "Чтобы активировать окружение в текущей сессии, выполните:"
echo "   poetry shell"
echo "или (для Windows Git Bash) используйте путь из 'poetry env info --path'"

echo ""
echo "setup_env.sh завершён успешно"