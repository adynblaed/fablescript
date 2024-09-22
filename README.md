# FableScript Codex

**FableScript** is a powerful and intuitive tool designed for worldbuilders, game masters, and developers to effortlessly generate detailed Dungeons & Dragons 5th Edition character sheets. Leveraging the advanced capabilities of the vLLM library and structured data schemas, FableScript ensures that every character is uniquely crafted and seamlessly integrated into your campaigns.

## Table of Contents

-   [Features](#features)
-   [Installation](#installation)
-   [Requirements](#requirements)
-   [Usage](#usage)
    -   [Command-Line Arguments](#command-line-arguments)
    -   [Examples](#examples)
-   [Output](#output)
-   [Developer Guide](#developer-guide)
-   [Contributing](#contributing)
-   [License](#license)

## Features

-   **Customizable Attributes:** Specify race, class, alignment, and more to tailor characters to your campaign needs.
-   **Backstory Generation:** Automatically create rich backstories to add depth to each character.
-   **Party Creation:** Generate a cohesive party with inter-character relationships for dynamic storytelling.
-   **Flexible Output Formats:** Choose between JSON and YAML for easy integration with other tools and platforms.
-   **Unique Filenames:** Organized output with versioning and seed-based naming to prevent conflicts.
-   **Enhanced Logging:** Control verbosity for detailed insights during generation.
-   **Schema Validation:** Ensure all character sheets adhere to predefined standards for consistency.
-   **Robust Error Handling:** Receive actionable feedback to troubleshoot and refine character generation.

## Installation

1.  **Clone the Repository:**

    bash
    Copy code

    `git clone https://github.com/yourusername/FableScript.git cd FableScript`

2.  **Create a Virtual Environment (Optional but Recommended):**

    bash
    Copy code

    `python3 -m venv venv source venv/bin/activate`

3.  **Install Dependencies:**

    bash
    Copy code

    `pip install -r requirements.txt`


## Requirements

-   Python 3.8+
-   [vLLM](https://github.com/vllm-project/vllm)
-   [Outlines](https://github.com/outlines/outlines)
-   Pydantic >= 2.0
-   PyYAML
-   tqdm

Install all requirements using:

bash
Copy code

`pip install vllm outlines pydantic>=2.0 pyyaml tqdm`

## Usage

FableScript is operated via the command line, offering a range of options to customize character generation.

### Command-Line Arguments

-   `--prompt`: _(str)_ Custom prompt to define the character's theme and concept.
-   `--sheets`: _(int)_ Number of character sheets to generate.
-   `--name`: _(str)_ Name of the character(s) and base filename for outputs.
-   `--include-backstory`: _(flag)_ Include a generated backstory for each character.
-   `--race`: _(str)_ Desired race of the character(s).
-   `--class`: _(str)_ Desired class of the character(s).
-   `--alignment`: _(str)_ Desired alignment of the character(s).
-   `--party-size`: _(int)_ Number of characters to generate in a party.
-   `--format`: _(str)_ Output format (`json` or `yaml`). Default is `json`.
-   `--verbose`: _(flag)_ Enable detailed logging output.

### Examples

1.  **Generate a Single Character:**

    bash
    Copy code

    `python generate_character_sheet.py --prompt "A noble battlemage facing ascension" \     --sheets 1 --name "Alric Seersage" --include-backstory \     --race "Elf" --class "Wizard" --alignment "Neutral Good" \     --format json --verbose`

2.  **Create a Party of 4 Characters:**

    bash
    Copy code

    `python generate_character_sheet.py --prompt "Adventurers exploring ancient ruins" \     --sheets 4 --name "PartyMember" --include-backstory \     --race "Mixed" --class "Various" --alignment "Neutral" \     --party-size 4 --format yaml --verbose`

3.  **Generate Multiple Characters Without Backstories:**

    bash
    Copy code

    `python generate_character_sheet.py --prompt "Stealthy rogues from the shadows" \     --sheets 5 --name "Shadowblade" --race "Human" --class "Rogue" \     --alignment "Chaotic Neutral" --format json`


## Output

-   **Directory:** All generated character sheets are saved in the `character_sheets` directory.
-   **Filename Structure:** `<CharacterName>_v<Version>_<Seed>.<format>`
    -   Example: `Alric_Seersage_v1_123456789.json`
-   **Formats:** JSON and YAML files containing comprehensive character details, including attributes, skills, equipment, spells, backstory, and relationships.

## Developer Guide

For developers looking to extend or modify FableScript, the script is structured for ease of customization:

-   **Modifiable Globals:**

    -   `MODEL_NAME`: Change the language model used for generation.
    -   `OUTPUT_DIR`: Specify a different directory for output files.
    -   `MAX_RETRIES`: Adjust the number of generation attempts.
    -   `GENERATION_TEMPERATURE` & `BACKSTORY_TEMPERATURE`: Control creativity in text generation.
    -   `GENERATION_MAX_TOKENS` & `BACKSTORY_MAX_TOKENS`: Set token limits for generated content.
-   **Pydantic Models:**

    -   The `CharacterSheet` class defines the schema for character sheets. Modify fields or validation rules as needed.
-   **Logging:**

    -   Customize logging behavior by adjusting the `configure_logging` method within the `CharacterSheetGenerator` class.
-   **Extensions:**

    -   Integrate additional features such as exporting to other formats, enhancing backstory complexity, or interfacing with other RPG tools.

## Contributing

Contributions are welcome! Whether you're reporting bugs, suggesting features, or submitting pull requests, your input helps improve FableScript.

1.  **Fork the Repository**
2.  **Create a Feature Branch**

    bash
    Copy code

    `git checkout -b feature/YourFeature`

3.  **Commit Your Changes**
4.  **Push to the Branch**

    bash
    Copy code

    `git push origin feature/YourFeature`

5.  **Open a Pull Request**

Please ensure your contributions adhere to the project's coding standards and include relevant tests where applicable.

## License

This project is licensed under the MIT License.