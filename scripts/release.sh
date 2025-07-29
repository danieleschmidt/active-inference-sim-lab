#!/bin/bash
# Release automation script for active-inference-sim-lab
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="main"
PYTHON="python3"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    command -v git >/dev/null 2>&1 || { log_error "git is required but not installed."; exit 1; }
    command -v python3 >/dev/null 2>&1 || { log_error "python3 is required but not installed."; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required but not installed."; exit 1; }
    
    # Check Python packages
    python3 -c "import toml" 2>/dev/null || { log_error "toml package is required. Install with: pip install toml"; exit 1; }
    python3 -c "import twine" 2>/dev/null || { log_error "twine package is required. Install with: pip install twine"; exit 1; }
    
    log_success "All dependencies are available"
}

check_git_status() {
    log_info "Checking git status..."
    
    # Check if we're on the correct branch
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "$BRANCH" ]; then
        log_error "Must be on $BRANCH branch. Currently on $current_branch"
        exit 1
    fi
    
    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_error "There are uncommitted changes. Please commit or stash them first."
        git status --porcelain
        exit 1
    fi
    
    # Check if we're up to date with remote
    git fetch origin
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/$BRANCH)" ]; then
        log_error "Local branch is not up to date with origin/$BRANCH"
        exit 1
    fi
    
    log_success "Git status is clean"
}

get_current_version() {
    python3 -c "
import toml
config = toml.load('pyproject.toml')
print(config['project']['version'])
    "
}

bump_version() {
    local bump_type=$1
    local current_version=$2
    
    python3 - <<EOF
import re
import sys

def bump_version(version, bump_type):
    # Parse semantic version
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$', version)
    if not match:
        print(f"Invalid version format: {version}", file=sys.stderr)
        sys.exit(1)
    
    major, minor, patch, prerelease = match.groups()
    major, minor, patch = int(major), int(minor), int(patch)
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        print(f"Invalid bump type: {bump_type}", file=sys.stderr)
        sys.exit(1)
    
    return f"{major}.{minor}.{patch}"

print(bump_version("$current_version", "$bump_type"))
EOF
}

update_version_files() {
    local new_version=$1
    
    log_info "Updating version to $new_version..."
    
    # Update pyproject.toml
    python3 - <<EOF
import toml

# Read current config
with open('pyproject.toml', 'r') as f:
    config = toml.load(f)

# Update version
config['project']['version'] = '$new_version'

# Write back
with open('pyproject.toml', 'w') as f:
    toml.dump(config, f)
EOF
    
    # Update version in C++ CMakeLists.txt if it exists
    if [ -f "CMakeLists.txt" ]; then
        sed -i.bak "s/project(active_inference VERSION [0-9.]*/project(active_inference VERSION $new_version/" CMakeLists.txt
        rm CMakeLists.txt.bak
    fi
    
    log_success "Version files updated"
}

run_tests() {
    log_info "Running test suite..."
    
    # Run Python tests
    if ! make test-python; then
        log_error "Python tests failed"
        exit 1
    fi
    
    # Run C++ tests if available
    if [ -d "build" ] && [ -f "build/Makefile" ]; then
        if ! make test-cpp; then
            log_error "C++ tests failed"
            exit 1
        fi
    fi
    
    # Run linting
    if ! make lint; then
        log_error "Linting failed"
        exit 1
    fi
    
    log_success "All tests passed"
}

build_package() {
    log_info "Building package..."
    
    # Clean previous builds
    make clean
    
    # Build package
    if ! make build-package; then
        log_error "Package build failed"
        exit 1
    fi
    
    # Verify package
    if ! python3 -m twine check dist/*; then
        log_error "Package verification failed"
        exit 1
    fi
    
    log_success "Package built successfully"
}

update_changelog() {
    local version=$1
    local date=$(date +%Y-%m-%d)
    
    log_info "Please update CHANGELOG.md for version $version"
    log_warning "Manual step required: Add release notes to CHANGELOG.md"
    
    # Open changelog in editor if available
    if command -v $EDITOR >/dev/null 2>&1; then
        $EDITOR CHANGELOG.md
    elif command -v nano >/dev/null 2>&1; then
        nano CHANGELOG.md
    elif command -v vim >/dev/null 2>&1; then
        vim CHANGELOG.md
    else
        log_warning "No editor found. Please manually update CHANGELOG.md"
    fi
    
    read -p "Have you updated CHANGELOG.md? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "Please update CHANGELOG.md and run the script again"
        exit 1
    fi
}

create_git_tag() {
    local version=$1
    local tag="v$version"
    
    log_info "Creating git commit and tag..."
    
    # Add all changed files
    git add .
    
    # Create commit
    git commit -m "chore: bump version to $version

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create annotated tag
    git tag -a "$tag" -m "Release $version

$(grep -A 20 "## \[$version\]" CHANGELOG.md | tail -n +2 | head -n -1)"
    
    log_success "Created commit and tag $tag"
}

push_release() {
    local version=$1
    local tag="v$version"
    
    log_info "Pushing release to remote..."
    
    # Push commit and tag
    git push origin $BRANCH
    git push origin $tag
    
    log_success "Release pushed to remote"
}

create_github_release() {
    local version=$1
    local tag="v$version"
    
    if command -v gh >/dev/null 2>&1; then
        log_info "Creating GitHub release..."
        
        # Extract changelog content for this version
        changelog_content=$(grep -A 50 "## \[$version\]" CHANGELOG.md | tail -n +2 | sed '/^## \[/q' | head -n -1)
        
        gh release create "$tag" \
            --title "Release $version" \
            --notes "$changelog_content" \
            dist/*
        
        log_success "GitHub release created"
    else
        log_warning "GitHub CLI not found. Please create release manually at:"
        log_warning "https://github.com/terragon-labs/active-inference-sim-lab/releases/new?tag=$tag"
    fi
}

publish_pypi() {
    local version=$1
    
    read -p "Publish to PyPI? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Publishing to PyPI..."
        
        if ! python3 -m twine upload dist/*; then
            log_error "PyPI upload failed"
            exit 1
        fi
        
        log_success "Published to PyPI"
        log_info "Package available at: https://pypi.org/project/active-inference-sim-lab/$version/"
    else
        log_info "Skipping PyPI publication"
        log_info "To publish manually later, run: python3 -m twine upload dist/*"
    fi
}

main() {
    local bump_type=$1
    
    # Validate arguments
    if [ "$#" -ne 1 ] || [[ ! "$bump_type" =~ ^(patch|minor|major)$ ]]; then
        echo "Usage: $0 [patch|minor|major]"
        echo ""
        echo "  patch: Bug fixes and minor improvements (x.y.Z)"
        echo "  minor: New features and enhancements (x.Y.z)"  
        echo "  major: Breaking changes and major updates (X.y.z)"
        exit 1
    fi
    
    log_info "Starting $bump_type release..."
    
    # Pre-flight checks
    check_dependencies
    check_git_status
    
    # Get version information
    current_version=$(get_current_version)
    new_version=$(bump_version "$bump_type" "$current_version")
    
    log_info "Current version: $current_version"
    log_info "New version: $new_version"
    
    # Confirm with user
    read -p "Proceed with $bump_type release ($current_version -> $new_version)? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Release cancelled"
        exit 0
    fi
    
    # Release process
    update_version_files "$new_version"
    update_changelog "$new_version"
    run_tests
    build_package
    create_git_tag "$new_version"
    push_release "$new_version"
    create_github_release "$new_version"
    publish_pypi "$new_version"
    
    log_success "Release $new_version completed successfully!"
    log_info "Next steps:"
    log_info "  1. Monitor CI/CD pipeline for any issues"
    log_info "  2. Verify package installation: pip install active-inference-sim-lab==$new_version"
    log_info "  3. Update documentation if needed"
    log_info "  4. Announce release to community"
}

# Run main function with all arguments
main "$@"