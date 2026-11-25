#!/usr/bin/env bash
# Delete Windows "*:Zone.Identifier" files under a directory (default: repo root)
# and drop them from the git index if tracked.
# Usage: bash tools/remove_zone_identifier.sh [dir]
set -euo pipefail
root="${1:-$(git rev-parse --show-toplevel)}"
cd "$root"
git ls-files -z | grep -zE ':Zone\.Identifier$' | xargs -0 -r git rm --cached -q -- || true
n=$(find . -name '*:Zone.Identifier' -not -path './.git/*' | wc -l)
find . -name '*:Zone.Identifier' -not -path './.git/*' -delete
echo "removed $n Zone.Identifier file(s)"
