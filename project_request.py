import argparse
import re
import requests
import sys
from typing import Tuple, Dict, List, Any, Optional
from urllib.parse import urlparse


def parse_github_url(url: str) -> Tuple[str, str]:
    """Extract owner and repository name from a GitHub URL.
    
    Args:
        url: A GitHub repository URL (e.g., https://github.com/owner/repo)
        
    Returns:
        A tuple containing (owner, repo) strings
        
    Raises:
        ValueError: If the URL is not a valid GitHub repository URL
    """
    parsed_url = urlparse(url)
    if 'github.com' not in parsed_url.netloc:
        raise ValueError(f"Not a GitHub URL: {url}")
    
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    owner, repo = path_parts[0], path_parts[1]
    return owner, repo


def get_repo_info(owner: str, repo: str) -> Dict[str, Any]:
    """Fetch repository information from GitHub API.
    
    Args:
        owner: GitHub repository owner/organization name
        repo: GitHub repository name
        
    Returns:
        Dictionary containing repository information from GitHub API
        
    Raises:
        Exception: If the API request fails
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch repository info: {response.status_code} {response.text}")
        
    return response.json()


def generate_entry(repo_info: Dict[str, Any], repo_url: str, project_name: Optional[str]=None) -> Tuple[str, str]:
    """Generate formatted entry for README.md.
    
    Args:
        repo_info: Repository information from GitHub API
        repo_url: GitHub repository URL
        project_name: The project name
        
    Returns:
        A tuple containing:
            - project_name: The name of the project
            - entry: A formatted markdown string for the README entry
    """
    description = repo_info.get('description', '')
    
    # Extract owner and repo for shields.io URLs
    owner, repo = parse_github_url(repo_url)
    
    # Generate shields.io URLs
    stars_badge = f"![Stars](https://img.shields.io/github/stars/{owner}/{repo}.svg?style=flat&color=green)"
    contributors_badge = f"![Contributors](https://img.shields.io/github/contributors/{owner}/{repo}?color=green)"
    last_commit_badge = f"![LastCommit](https://img.shields.io/github/last-commit/{owner}/{repo}?color=green)"
    
    # Format the entry
    entry = f"* **[{project_name}]({repo_url})**: {description} {stars_badge} {contributors_badge} {last_commit_badge}"
    
    return project_name, entry


def find_section(content: str, section_name: str) -> Tuple[int, int, List[str]]:
    """Find the specified section in the README content.
    
    Args:
        content: The full content of the README.md file
        section_name: The name of the section to find (e.g., "framework")
        
    Returns:
        A tuple containing:
            - section_start_line: The line number where the section starts
            - section_end_line: The line number where the section ends
            - lines: List of all lines in the content
            
    Raises:
        ValueError: If the specified section is not found
    """
    
    # Define patterns for different section levels
    section_patterns = [
        # Main section (##)
        re.compile(r'##\s+([^\n]+)'),
        # Subsection (###)
        re.compile(r'###\s+([^\n]+)')
    ]
    
    lines = content.split('\n')
    section_start_line = -1
    section_end_line = -1
    current_section = ""
    
    for i, line in enumerate(lines):
        # Check if this line starts a section
        for pattern in section_patterns:
            match = pattern.match(line)
            if match:
                # If we already found our section, this new section marks the end
                if section_start_line != -1 and section_end_line == -1:
                    section_end_line = i
                    break
                
                # Check if this is the section we're looking for
                current_section = match.group(1).strip().lower()
                if current_section == section_name:
                    section_start_line = i
                    break
    
    # If we found the start but not the end, the section goes to the end of the file
    if section_start_line != -1 and section_end_line == -1:
        section_end_line = len(lines)
    
    if section_start_line == -1:
        raise ValueError(f"Section '{section_name}' not found in README.md")
    
    return section_start_line, section_end_line, lines


def insert_entry(lines: List[str], section_start_line: int, section_end_line: int, project_name: str, new_entry: str) -> List[str]:
    """Insert the new entry in alphabetical order within the section.
    
    Args:
        lines: List of all lines in the README.md file
        section_start_line: The line number where the section starts
        section_end_line: The line number where the section ends
        project_name: The name of the project
        new_entry: The formatted entry to insert
        
    Returns:
        Updated list of lines with the new entry inserted in alphabetical order
    """
    
    # Find the correct position to insert the new entry
    insert_position = section_end_line
    
    # Skip the section header
    for i in range(section_start_line + 1, section_end_line):
        line = lines[i]
        # Check if this line is an entry
        entry_name_match = re.search(r'\*\s+\*\*\[([^\]]+)\]', line)
        if entry_name_match:
            entry_name = entry_name_match.group(1).lower()
            # If the new entry comes before this entry alphabetically (case-insensitive comparison)
            if project_name.lower() < entry_name:
                insert_position = i
                break
    
    # Insert the new entry at the determined position
    lines.insert(insert_position, new_entry)
    
    return lines


def update_readme(readme_path: str, section_name: str, project_name: str, new_entry: str) -> bool:
    """Update the README.md file with the new entry.
    
    Args:
        readme_path: Path to the README.md file
        section_name: The name of the section to add the entry to
        project_name: The name of the project
        new_entry: The formatted entry to add
        
    Returns:
        True if the update was successful, False otherwise
    """

    try:
        with open(readme_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        section_start_line, section_end_line, lines = find_section(content, section_name)
        updated_lines = insert_entry(lines, section_start_line, section_end_line, project_name, new_entry)
        
        # Write the updated content back to the file
        with open(readme_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(updated_lines))
        
        return True
    
    except Exception as e:
        print(f"Error updating README.md: {str(e)}")
        return False


def main() -> None:
    """Main function to parse arguments and execute the script.
    
    Command line arguments:
        --section/-s: The section to add the project to (e.g., "framework")
        --repo_url/-r: The GitHub repository URL
        --name/-n: Custom project name

    Example:
        python project_request.py --section framework --repo_url https://github.com/google/adk-python --name "Agent Development Kit (ADK)"
    """
    parser = argparse.ArgumentParser(description='Add a new project to the README.md file.')
    parser.add_argument('--section', '-s', required=True, help='The section to add the project to (e.g., "framework")')
    parser.add_argument('--repo_url', '-r', required=True, help='The GitHub repository URL')
    parser.add_argument('--name', '-n', required=True, help='Custom project name')
    
    args = parser.parse_args()
    
    try:
        # Parse GitHub URL
        owner, repo = parse_github_url(args.repo_url)
        
        # Get repository information
        repo_info = get_repo_info(owner, repo)
        
        # Generate the entry
        project_name, entry = generate_entry(repo_info, args.repo_url, args.name)
        
        # Update the README.md file
        readme_path = 'README.md'
        success = update_readme(readme_path, args.section.lower(), project_name, entry)
        
        if not success:
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    print(f"Successfully added '{project_name}' to the {args.section} section.")


if __name__ == '__main__':
    main()
