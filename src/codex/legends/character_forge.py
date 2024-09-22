#!/usr/bin/env python3
"""
LegendSmith Character Forge v4.0

This script generates comprehensive and customizable character sheets for Dungeons & Dragons 5th Edition,
leveraging the Outlines library for structured and consistent JSON/YAML outputs. It ensures adherence
to predefined schemas using Pydantic V2 models and offers extensive configuration options for scalability
and maintainability.

Features:
- Advanced character customization options
- Efficient and structured JSON generation using Outlines and Pydantic V2 models
- Integration of vLLM via Outlines for improved performance
- Balance assurance mechanisms to ensure game compliance
- Import and export functionality for character sheets
- Enhanced utilization of Outlines for consistency and reliability
- YAML configuration support for scalable settings
- Comprehensive logging with configurable verbosity
- Robust error handling and validation
- Modular architecture for easy maintenance and scalability
- Rejected character sheets are saved when verbose mode is enabled
- All logs are saved to a .log file with detailed information

Requirements:
- Python 3.8+
- outlines
- pydantic>=2.0.0
- pyyaml
- tqdm
- python-dotenv
- vllm
- networkx (for relationship mapping)
- matplotlib (for graph visualization)

Usage:
    python character_forge.py --prompt "A noble battlemage facing ascension" \
        --sheets 5 --name "Alric Seersage" --include-backstory \
        --race "Elf" --class "Wizard" --alignment "Neutral Good" \
        --level 10 --feats "Spell Sniper, War Caster" \
        --skills "Arcana, History" --equipment "Staff of Power, Robes" \
        --spells "Fireball, Teleport" --format json --verbose --config config.yaml

Author: Your Name
Version: 4.0
"""

import json
import argparse
import random
import logging
import os
import sys
from typing import List, Optional, Callable, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, ValidationInfo, conint, constr
from outlines import models, generate
from vllm import SamplingParams
import yaml
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables from .env file if present
load_dotenv()

# -----------------------------
# Configuration Module
# -----------------------------

class SamplingConfig(BaseModel):
    n: int = 1
    best_of: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    ignore_eos: bool = False
    max_tokens: int = 1024
    min_tokens: int = 0
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None

class Config(BaseModel):
    MODEL_NAME: str = Field(
        default="NousResearch/Meta-Llama-2-7b-chat-hf",
        description="Language model identifier"
    )
    OUTPUT_DIR: str = Field(
        default="character_sheets",
        description="Directory to save character sheets"
    )
    MAX_RETRIES: int = Field(
        default=5,
        description="Max retries for character generation"
    )
    GENERATE_BACKSTORY: bool = Field(
        default=True,
        description="Enable backstory generation"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    character_generation_params: SamplingConfig = Field(default_factory=lambda: SamplingConfig(
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9
    ))
    backstory_generation_params: SamplingConfig = Field(default_factory=lambda: SamplingConfig(
        temperature=0.5,
        max_tokens=512,
        top_p=0.9
    ))
    relationship_generation_params: SamplingConfig = Field(default_factory=lambda: SamplingConfig(
        temperature=0.3,
        max_tokens=150,
        top_p=0.8
    ))
    BALANCE_CHECKS: bool = Field(
        default=True,
        description="Enable balance assurance checks"
    )

    class Config:
        env_prefix = "CHAR_GEN_"

    @field_validator('LOG_LEVEL')
    def validate_log_level(cls, v: str, info: ValidationInfo) -> str:
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in levels:
            raise ValueError(f"LOG_LEVEL must be one of {levels}")
        return v.upper()

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.debug(f"YAML configuration loaded from {config_path}.")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        return {}

def merge_configs(default: Dict[str, Any], yaml_conf: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges default, YAML, and CLI configurations with priority: CLI > YAML > Default.
    """
    merged = default.copy()
    # Update with YAML config
    for key, value in yaml_conf.items():
        if key in merged:
            merged[key] = value
    # Update with CLI args (non-None)
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged

# -----------------------------
# Logging Configuration
# -----------------------------

def configure_logging(level: str, output_dir: str, verbose: bool) -> logging.Logger:
    """
    Configures the logging level and format.
    """
    logger = logging.getLogger("CharacterForge")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    # Ensure the output directory exists for the log file
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = getattr(logging, level.upper(), logging.INFO)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file_path = os.path.join(output_dir, 'character_forge.log')
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_level = logging.DEBUG  # Always log DEBUG level to file
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# -----------------------------
# Data Models Module
# -----------------------------

class Spell(BaseModel):
    name: str
    level: conint(ge=0, le=9) = Field(..., description="Spell level (0 for cantrips)")
    description: Optional[str] = Field(None, description="Description of the spell")
    properties: Optional[Dict[str, Any]] = None

class CharacterSheet(BaseModel):
    name: str = Field(..., description="Character's name", max_length=50)
    race: str = Field(..., description="Character's race", max_length=30)
    character_class: str = Field(..., alias='class', description="Character's class", max_length=30)
    level: conint(ge=1, le=20) = Field(1, description="Character's level")
    background: str = Field(..., description="Character's background")
    alignment: str = Field(..., description="Character's alignment", max_length=20)
    strength: conint(ge=1, le=20) = Field(..., description="Strength ability score")
    dexterity: conint(ge=1, le=20) = Field(..., description="Dexterity ability score")
    constitution: conint(ge=1, le=20) = Field(..., description="Constitution ability score")
    intelligence: conint(ge=1, le=20) = Field(..., description="Intelligence ability score")
    wisdom: conint(ge=1, le=20) = Field(..., description="Wisdom ability score")
    charisma: conint(ge=1, le=20) = Field(..., description="Charisma ability score")
    feats: Optional[List[str]] = Field(None, description="List of character's feats")
    skills: Optional[List[str]] = Field(None, description="List of character's skills")
    proficiencies: Optional[List[str]] = Field(None, description="List of character's proficiencies")
    equipment: Optional[List[str]] = Field(None, description="List of character's equipment")
    spells: Optional[List[Spell]] = Field(None, description="List of character's known spells (if applicable)")
    hit_points: conint(ge=1, le=300) = Field(..., description="Character's maximum hit points")
    armor_class: conint(ge=10, le=40) = Field(..., description="Character's armor class")
    personality_traits: Optional[str] = Field(None, description="Character's personality traits")
    ideals: Optional[str] = Field(None, description="Character's ideals")
    bonds: Optional[str] = Field(None, description="Character's bonds")
    flaws: Optional[str] = Field(None, description="Character's flaws")
    backstory: Optional[str] = Field(None, description="Character's backstory")
    relationships: Optional[List[str]] = Field(None, description="Relationships with other characters")
    seed: int = Field(..., description="Random seed used for generating the character")

    @field_validator('spells', mode='before')
    def validate_spells(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, list):
            return [Spell(**spell) if isinstance(spell, dict) else spell for spell in v]
        return v

    @field_validator('relationships', mode='before')
    def validate_relationships(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, list):
            return v
        return []

# -----------------------------
# Utilities Module
# -----------------------------

def load_backstory(character: CharacterSheet, generator: Callable, sampling_params: SamplingParams) -> str:
    """
    Generates a backstory for the given character using the text generator.
    """
    prompt = (
        f"Provide a detailed backstory for {character.name}, "
        f"a level {character.level} {character.race} {character.character_class} with alignment {character.alignment}. "
        f"Include their feats: {', '.join(character.feats or [])}, skills: {', '.join(character.skills or [])}, "
        f"and equipment: {', '.join(character.equipment or [])}. "
        f"Include their motivations, significant life events, and personal relationships."
    )
    logging.getLogger("CharacterForge").debug(f"Generating backstory with prompt: {prompt}")
    try:
        backstory = generator(prompt, sampling_params=sampling_params).strip()
        logging.getLogger("CharacterForge").debug("Backstory generation completed successfully.")
        return backstory
    except Exception as e:
        logging.getLogger("CharacterForge").error(f"Failed to generate backstory: {e}")
        return "No backstory available."

def generate_filename(base_name: str, version: int, seed: int, output_format: str) -> str:
    """
    Generates a unique filename based on base name, version, and seed.
    """
    safe_base = "_".join(base_name.split()).strip("_")
    filename = f"{safe_base}_v{version}_{seed}.{output_format}"
    # Ensure the filename is safe
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.')).rstrip()
    return filename

def assign_relationships(characters: List[CharacterSheet], generator: Callable, sampling_params: SamplingParams):
    """
    Assigns inter-character relationships within a party using semantic generation and visualizes them.
    """
    logger = logging.getLogger("CharacterForge")
    logger.info("Generating inter-character relationships...")
    relationship_graph = nx.DiGraph()
    for character in characters:
        relationship_graph.add_node(character.name)
    for character in characters:
        other_characters = [c.name for c in characters if c.name != character.name]
        if other_characters:
            chosen = random.choice(other_characters)
            prompt = (
                f"Describe the relationship between {character.name} and {chosen}. "
                f"Include the nature of their bond and any shared history."
            )
            try:
                relationship = generator(prompt, sampling_params=sampling_params).strip()
                if character.relationships:
                    character.relationships.append(relationship)
                else:
                    character.relationships = [relationship]
                relationship_graph.add_edge(character.name, chosen, relationship=relationship)
                logger.debug(f"Assigned relationship for {character.name}: {relationship}")
            except Exception as e:
                logger.error(f"Failed to generate relationship for {character.name}: {e}")
    # Visualize the relationship graph
    try:
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(relationship_graph)
        nx.draw(relationship_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=10)
        edge_labels = nx.get_edge_attributes(relationship_graph, 'relationship')
        nx.draw_networkx_edge_labels(relationship_graph, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Character Relationship Map")
        plt.axis('off')
        plt.savefig(os.path.join(characters[0].seed_dir, 'relationships.png'))
        plt.close()
        logger.info("Relationship map saved successfully.")
    except Exception as e:
        logger.error(f"Failed to visualize relationship map: {e}")

def import_character_sheet(filepath: str) -> Optional[CharacterSheet]:
    """
    Imports a character sheet from a JSON or YAML file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.lower().endswith('.yaml') or filepath.lower().endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
            character = CharacterSheet(**data)
            return character
    except Exception as e:
        logging.getLogger("CharacterForge").error(f"Failed to import character sheet from {filepath}: {e}")
        return None

# -----------------------------
# Balance Assurance Module
# -----------------------------

def check_balance(character: CharacterSheet) -> bool:
    """
    Checks if the character is balanced according to game mechanics.
    """
    logger = logging.getLogger("CharacterForge")
    balance_issues = []

    # Example balance checks
    max_skill_points = character.level + 3  # Simplified example
    if character.skills and len(character.skills) > max_skill_points:
        balance_issues.append(f"Too many skills selected: {len(character.skills)} (max {max_skill_points})")

    max_feats = character.level // 2  # Simplified example
    if character.feats and len(character.feats) > max_feats:
        balance_issues.append(f"Too many feats selected: {len(character.feats)} (max {max_feats})")

    # Check ability scores total
    ability_scores = [
        character.strength,
        character.dexterity,
        character.constitution,
        character.intelligence,
        character.wisdom,
        character.charisma
    ]
    total_ability_points = sum(ability_scores)
    if total_ability_points > 75:  # Arbitrary total for balance
        balance_issues.append(f"Total ability scores too high: {total_ability_points} (max 75)")

    if balance_issues:
        logger.warning(f"Balance issues found for {character.name}: {balance_issues}")
        return False
    else:
        logger.debug(f"Character {character.name} passed balance checks.")
        return True

# -----------------------------
# Character Forge Class
# -----------------------------

class CharacterForge:
    """
    A class to generate LegendSmith character sheets using Outlines and generative AI.
    """

    def __init__(self, config: Config, logger: logging.Logger, verbose: bool):
        """
        Initializes the CharacterForge by loading the language model and creating generators.
        """
        self.config = config
        self.logger = logger
        self.verbose = verbose
        self.model = self.load_language_model()
        self.generator = self.create_json_generator()
        self.text_generator = self.create_text_generator()
        # Create 'rejected' directory if verbose is enabled
        if self.verbose:
            self.rejected_dir = os.path.join(self.config.OUTPUT_DIR, 'rejected')
            Path(self.rejected_dir).mkdir(parents=True, exist_ok=True)

    def load_language_model(self) -> Optional[Callable]:
        """
        Loads the specified language model using Outlines with vLLM.
        """
        try:
            self.logger.debug(f"Loading language model '{self.config.MODEL_NAME}' with Outlines (vLLM)...")
            model = models.vllm(self.config.MODEL_NAME, trust_remote_code=True)
            self.logger.info("Language model loaded successfully.")
            return model
        except Exception as e:
            self.logger.critical(f"Failed to load the language model '{self.config.MODEL_NAME}': {e}")
            sys.exit(1)

    def create_json_generator(self) -> Callable:
        """
        Creates a JSON generator based on the CharacterSheet schema.
        """
        try:
            self.logger.debug("Creating JSON generator based on CharacterSheet schema...")
            generator = generate.json(self.model, CharacterSheet)
            self.logger.info("JSON generator created successfully.")
            return generator
        except Exception as e:
            self.logger.critical(f"Failed to create JSON generator: {e}")
            sys.exit(1)

    def create_text_generator(self) -> Callable:
        """
        Creates a text generator for generating backstories and relationships.
        """
        try:
            self.logger.debug("Creating text generator for backstory and relationships...")
            generator = generate.text(self.model)
            self.logger.info("Text generator created successfully.")
            return generator
        except Exception as e:
            self.logger.critical(f"Failed to create text generator: {e}")
            sys.exit(1)

    def generate_character_sheet(
        self,
        prompt: str,
        character_name: Optional[str] = None,
        desired_race: Optional[str] = None,
        desired_class: Optional[str] = None,
        desired_alignment: Optional[str] = None,
        desired_level: Optional[int] = None,
        desired_feats: Optional[List[str]] = None,
        desired_skills: Optional[List[str]] = None,
        desired_equipment: Optional[List[str]] = None,
        desired_spells: Optional[List[str]] = None,
        include_backstory: bool = False,
        seed: Optional[int] = None,
        attempt_number: int = 1
    ) -> Optional[CharacterSheet]:
        """
        Generates a character sheet using the provided model and generator.
        """
        raw_output = None  # Initialize raw_output to None
        try:
            # Define sampling parameters
            sampling_params = SamplingParams(
                n=self.config.character_generation_params.n,
                best_of=self.config.character_generation_params.best_of,
                temperature=self.config.character_generation_params.temperature,
                top_p=self.config.character_generation_params.top_p,
                frequency_penalty=self.config.character_generation_params.frequency_penalty,
                presence_penalty=self.config.character_generation_params.presence_penalty,
                repetition_penalty=self.config.character_generation_params.repetition_penalty,
                top_k=self.config.character_generation_params.top_k,
                min_p=self.config.character_generation_params.min_p,
                use_beam_search=self.config.character_generation_params.use_beam_search,
                length_penalty=self.config.character_generation_params.length_penalty,
                early_stopping=self.config.character_generation_params.early_stopping,
                ignore_eos=self.config.character_generation_params.ignore_eos,
                max_tokens=self.config.character_generation_params.max_tokens,
                min_tokens=self.config.character_generation_params.min_tokens,
                stop=self.config.character_generation_params.stop_sequences if self.config.character_generation_params.stop_sequences else None,
                seed=seed
            )

            # Generate the character sheet
            raw_output = self.generator(
                prompt,
                sampling_params=sampling_params,
                max_tokens=self.config.character_generation_params.max_tokens
            )
            self.logger.debug(f"Raw output on attempt {attempt_number}: {raw_output}")

            # Prepare updates based on provided arguments
            updates = {}
            if character_name:
                updates['name'] = character_name
            if desired_race:
                updates['race'] = desired_race
            if desired_class:
                updates['character_class'] = desired_class
            if desired_alignment:
                updates['alignment'] = desired_alignment
            if desired_level:
                updates['level'] = desired_level
            if desired_feats:
                updates['feats'] = desired_feats
            if desired_skills:
                updates['skills'] = desired_skills
            if desired_equipment:
                updates['equipment'] = desired_equipment
            if desired_spells:
                updates['spells'] = [Spell(name=spell_name, level=0) for spell_name in desired_spells]
            if seed is not None:
                updates['seed'] = seed

            # Apply updates if any
            if updates:
                for key, value in updates.items():
                    setattr(raw_output, key, value)

            # Perform balance checks if enabled
            if self.config.BALANCE_CHECKS:
                if not check_balance(raw_output):
                    raise ValidationError("Character failed balance checks.")

            # Generate backstory if enabled
            if include_backstory:
                backstory_sampling_params = SamplingParams(
                    n=self.config.backstory_generation_params.n,
                    best_of=self.config.backstory_generation_params.best_of,
                    temperature=self.config.backstory_generation_params.temperature,
                    top_p=self.config.backstory_generation_params.top_p,
                    frequency_penalty=self.config.backstory_generation_params.frequency_penalty,
                    presence_penalty=self.config.backstory_generation_params.presence_penalty,
                    repetition_penalty=self.config.backstory_generation_params.repetition_penalty,
                    top_k=self.config.backstory_generation_params.top_k,
                    min_p=self.config.backstory_generation_params.min_p,
                    use_beam_search=self.config.backstory_generation_params.use_beam_search,
                    length_penalty=self.config.backstory_generation_params.length_penalty,
                    early_stopping=self.config.backstory_generation_params.early_stopping,
                    ignore_eos=self.config.backstory_generation_params.ignore_eos,
                    max_tokens=self.config.backstory_generation_params.max_tokens,
                    min_tokens=self.config.backstory_generation_params.min_tokens,
                    stop=self.config.backstory_generation_params.stop_sequences if self.config.backstory_generation_params.stop_sequences else None,
                    seed=random.randint(1, int(1e9))
                )
                backstory = load_backstory(
                    raw_output,
                    self.text_generator,
                    sampling_params=backstory_sampling_params
                )
                raw_output.backstory = backstory

            self.logger.debug(f"Character sheet validation succeeded for {raw_output.name}.")
            return raw_output
        except ValidationError as ve:
            # Log detailed validation errors
            error_details = str(ve)
            self.logger.warning(f"Validation error on attempt {attempt_number}: {error_details}")
            # Save rejected output if verbose mode is enabled
            if self.verbose:
                self.save_rejected_output(raw_output, seed, attempt_number, ve)
        except Exception as e:
            self.logger.error(f"Error during generation on attempt {attempt_number}: {e}")
            # Save rejected output if verbose mode is enabled
            if self.verbose:
                self.save_rejected_output(raw_output, seed, attempt_number, e)
        return None

    def save_rejected_output(self, output: Any, seed: int, attempt_number: int, error: Exception):
        """
        Saves rejected outputs to the 'rejected' folder when verbose mode is enabled.
        """
        try:
            filename = f"rejected_{seed}_attempt{attempt_number}.txt"
            filepath = os.path.join(self.rejected_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Error: {error}\n\n")
                f.write("Raw Output:\n")
                if output is not None:
                    # If output is an object, serialize it to JSON
                    if isinstance(output, BaseModel):
                        json_output = output.model_dump(mode='json', by_alias=True)
                        json.dump(json_output, f, indent=2, ensure_ascii=False)
                    else:
                        f.write(str(output))
                else:
                    f.write("No output generated.")
            self.logger.debug(f"Rejected output saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save rejected output: {e}")

    def save_character_sheet(self, character: CharacterSheet, filepath: str, output_format: str) -> bool:
        """
        Saves the character sheet to a file and validates the saved file.
        """
        try:
            # Create a directory for each character based on seed
            character.seed_dir = os.path.join(self.config.OUTPUT_DIR, f"{character.name}_{character.seed}")
            Path(character.seed_dir).mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(character.seed_dir, os.path.basename(filepath))

            with open(filepath, 'w', encoding='utf-8') as f:
                if output_format.lower() == 'yaml':
                    yaml.dump(character.model_dump(mode='json', by_alias=True), f, sort_keys=False, allow_unicode=True)
                else:
                    json.dump(character.model_dump(mode='json', by_alias=True), f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Character sheet written to {filepath}.")

            # Validate the saved file
            with open(filepath, 'r', encoding='utf-8') as f:
                if output_format.lower() == 'yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
                CharacterSheet(**data)
            self.logger.debug(f"Validation of saved file {filepath} succeeded.")
            return True
        except (IOError, OSError) as e:
            self.logger.error(f"I/O error while saving file {filepath}: {e}")
        except ValidationError as ve:
            self.logger.error(f"Validation error for saved file {filepath}: {ve}")
        except Exception as e:
            self.logger.error(f"Unexpected error while saving file {filepath}: {e}")
        return False

    def run(
        self,
        prompt: str,
        sheets: int,
        name: Optional[str],
        include_backstory: bool,
        race: Optional[str],
        character_class: Optional[str],
        alignment: Optional[str],
        level: Optional[int],
        feats: Optional[List[str]],
        skills: Optional[List[str]],
        equipment: Optional[List[str]],
        spells: Optional[List[str]],
        party_size: int,
        output_format: str,
        import_file: Optional[str] = None
    ):
        """
        Executes the character sheet generation process based on the provided parameters.
        """
        self.logger.info("Starting LegendSmith Character Forge...")

        # Base prompt to guide the model in generating the character sheet
        base_prompt = (
            "As an expert in D&D 5th Edition character creation, generate a complete character sheet in JSON format. "
            "Ensure the JSON is valid, well-formatted, and includes all required fields as per the schema provided. "
            "The character sheet should include the character's name, race, class, level (1-20), background, alignment, "
            "ability scores (strength, dexterity, constitution, intelligence, wisdom, charisma) ranging from 1 to 20, "
            "feats, skills, proficiencies, equipment, spells (if applicable), hit points (appropriate for level and class), "
            "armor class (10 to 40), personality traits, ideals, bonds, flaws, and optionally a backstory. "
            "All numeric values must be within valid D&D 5e ranges. Do not include any extra text or explanation."
        )

        # If a custom prompt is provided, append it to the base prompt
        if prompt:
            full_prompt = f"{base_prompt}\n{prompt}"
            self.logger.debug("Custom prompt provided and appended to the base prompt.")
        else:
            full_prompt = base_prompt
            self.logger.debug("No custom prompt provided. Using base prompt.")

        # Determine the number of sheets to generate
        total_sheets = sheets if party_size == 0 else party_size
        if party_size > 0:
            self.logger.debug(f"Party size specified: {party_size}. Overriding sheets to generate.")
        else:
            self.logger.debug(f"Sheets specified: {sheets}.")

        # Import character sheet if provided
        imported_character = None
        if import_file:
            imported_character = import_character_sheet(import_file)
            if imported_character:
                self.logger.info(f"Imported character sheet from {import_file}")

        # Generate the specified number of character sheets with an enhanced progress bar
        characters = []
        self.logger.info(f"Generating {total_sheets} character sheet(s)...")
        for i in tqdm(range(1, total_sheets + 1), desc="Generating Characters", unit="sheet", dynamic_ncols=True):
            self.logger.debug(f"Generating character sheet #{i}...")
            # Define a unique seed for each character
            seed = random.randint(1, int(1e9))
            self.logger.debug(f"Generated seed for character #{i}: {seed}")

            # Determine character name
            if party_size > 0:
                character_name = f"{name}_{i}" if name else f"PartyMember_{i}"
            else:
                character_name = name if name else f"Character_{i}"
            self.logger.debug(f"Character name for sheet #{i}: {character_name}")

            # Attempt to generate the character sheet
            character = None
            for attempt in range(1, self.config.MAX_RETRIES + 1):
                self.logger.debug(f"Attempt {attempt} to generate character sheet.")
                character = self.generate_character_sheet(
                    prompt=full_prompt,
                    character_name=character_name,
                    desired_race=race,
                    desired_class=character_class,
                    desired_alignment=alignment,
                    desired_level=level,
                    desired_feats=feats,
                    desired_skills=skills,
                    desired_equipment=equipment,
                    desired_spells=spells,
                    include_backstory=include_backstory,
                    seed=seed,
                    attempt_number=attempt
                )
                if character:
                    break  # Exit retry loop if character generation is successful

            if character is None:
                self.logger.warning(f"Failed to generate character sheet #{i}.")
                continue

            characters.append(character)
            self.logger.info(f"Generated character: {character.name} (Seed: {character.seed})")

        # Generate relationships if generating a party
        if party_size > 1 and len(characters) > 1:
            self.logger.debug("Generating relationships among party members.")
            relationship_sampling_params = SamplingParams(
                n=self.config.relationship_generation_params.n,
                best_of=self.config.relationship_generation_params.best_of,
                temperature=self.config.relationship_generation_params.temperature,
                top_p=self.config.relationship_generation_params.top_p,
                frequency_penalty=self.config.relationship_generation_params.frequency_penalty,
                presence_penalty=self.config.relationship_generation_params.presence_penalty,
                repetition_penalty=self.config.relationship_generation_params.repetition_penalty,
                top_k=self.config.relationship_generation_params.top_k,
                min_p=self.config.relationship_generation_params.min_p,
                use_beam_search=self.config.relationship_generation_params.use_beam_search,
                length_penalty=self.config.relationship_generation_params.length_penalty,
                early_stopping=self.config.relationship_generation_params.early_stopping,
                ignore_eos=self.config.relationship_generation_params.ignore_eos,
                max_tokens=self.config.relationship_generation_params.max_tokens,
                min_tokens=self.config.relationship_generation_params.min_tokens,
                stop=self.config.relationship_generation_params.stop_sequences if self.config.relationship_generation_params.stop_sequences else None,
                seed=random.randint(1, int(1e9))
            )
            assign_relationships(
                characters,
                self.text_generator,
                sampling_params=relationship_sampling_params
            )

        # Save character sheets with an enhanced progress bar
        try:
            Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Output directory '{self.config.OUTPUT_DIR}' is ready.")
        except Exception as e:
            self.logger.critical(f"Failed to create output directory '{self.config.OUTPUT_DIR}': {e}")
            sys.exit(1)

        self.logger.info(f"Saving {len(characters)} character sheet(s) to '{self.config.OUTPUT_DIR}'...")
        with tqdm(total=len(characters), desc="Saving Characters", unit="sheet", dynamic_ncols=True) as pbar:
            for idx, character in enumerate(characters, start=1):
                # Generate unique filename with version and seed
                filename = generate_filename(
                    base_name=character.name,
                    version=idx,
                    seed=character.seed,
                    output_format=output_format
                )

                filepath = os.path.join(self.config.OUTPUT_DIR, filename)
                self.logger.debug(f"Saving character sheet to '{filepath}'.")

                # Save the character sheet
                success = self.save_character_sheet(character, filepath, output_format=output_format)
                if success:
                    self.logger.info(f"Character sheet saved to {filepath}")
                else:
                    self.logger.error(f"Failed to save character sheet for {character.name}")
                pbar.update(1)

        self.logger.info("Character sheet generation completed.")

# -----------------------------
# Argument Parsing Module
# -----------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="LegendSmith Character Forge v4.0")
    parser.add_argument('--prompt', type=str, required=False, default='',
                        help='Custom prompt for character theme and concept.')
    parser.add_argument('--sheets', type=int, required=False, default=1,
                        help='Number of character sheets to generate.')
    parser.add_argument('--name', type=str, required=False,
                        help='Name of the character (and base filename for output files).')
    parser.add_argument('--include-backstory', action='store_true',
                        help='Include a generated backstory for each character.')
    parser.add_argument('--race', type=str, required=False,
                        help='Desired race of the character(s).')
    parser.add_argument('--class', dest='character_class', type=str, required=False,
                        help='Desired class of the character(s).')
    parser.add_argument('--alignment', type=str, required=False,
                        help='Desired alignment of the character(s).')
    parser.add_argument('--level', type=int, required=False,
                        help='Desired level of the character(s).')
    parser.add_argument('--feats', type=str, nargs='*', required=False,
                        help='Desired feats of the character(s).')
    parser.add_argument('--skills', type=str, nargs='*', required=False,
                        help='Desired skills of the character(s).')
    parser.add_argument('--equipment', type=str, nargs='*', required=False,
                        help='Desired equipment of the character(s).')
    parser.add_argument('--spells', type=str, nargs='*', required=False,
                        help='Desired spells of the character(s).')
    parser.add_argument('--party-size', type=int, required=False, default=0,
                        help='Generate a party of characters with specified size.')
    parser.add_argument('--format', type=str, choices=['json', 'yaml'], default='json',
                        help='Output format (json or yaml).')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging and save rejected outputs.')
    parser.add_argument('--config', type=str, required=False, default='config.yaml',
                        help='Path to the YAML configuration file (default: config.yaml).')
    parser.add_argument('--import-file', type=str, required=False,
                        help='Path to an existing character sheet to import.')
    return parser.parse_args()

# -----------------------------
# Main Function
# -----------------------------

def main():
    """
    Main function to parse arguments and initiate character sheet generation.
    """
    args = parse_arguments()

    # Load YAML configuration
    yaml_config = load_yaml_config(args.config) if args.config else {}

    # Prepare CLI arguments as a dictionary
    cli_args = {
        'GENERATE_BACKSTORY': args.include_backstory if args.include_backstory else None,
        'LOG_LEVEL': 'DEBUG' if args.verbose else None,
    }

    # Merge configurations: Default < YAML < CLI
    default_config = Config().dict()
    merged_config_dict = merge_configs(default=default_config, yaml_conf=yaml_config, cli_args=cli_args)

    try:
        config = Config(**merged_config_dict)
    except ValidationError as ve:
        print(f"Configuration validation error: {ve}")
        sys.exit(1)

    # Ensure the output directory exists
    try:
        Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory '{config.OUTPUT_DIR}': {e}")
        sys.exit(1)

    # Configure logging based on merged configuration
    logger = configure_logging(config.LOG_LEVEL, config.OUTPUT_DIR, args.verbose)

    # Initialize the CharacterForge
    forge = CharacterForge(config=config, logger=logger, verbose=args.verbose)

    # Run the forge
    forge.run(
        prompt=args.prompt,
        sheets=args.sheets,
        name=args.name,
        include_backstory=args.include_backstory,
        race=args.race,
        character_class=args.character_class,
        alignment=args.alignment,
        level=args.level,
        feats=args.feats,
        skills=args.skills,
        equipment=args.equipment,
        spells=args.spells,
        party_size=args.party_size,
        output_format=args.format,
        import_file=args.import_file
    )

# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    main()
