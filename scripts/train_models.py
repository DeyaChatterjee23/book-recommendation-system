#!/usr/bin/env python3
"""
Script to train recommendation models.

This script handles the complete training pipeline:
1. Load and preprocess data
2. Train collaborative filtering model
3. Train content-based model
4. Train hybrid model
5. Save models and generate reports
"""

import asyncio
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from book_recommender.core.config import get_settings
from book_recommender.core.logger import setup_logging
from book_recommender.data.data_loader import data_loader
from book_recommender.data.data_preprocessor import data_preprocessor
from book_recommender.data.data_validator import data_validator
from book_recommender.models.collaborative_filtering import collaborative_model
from book_recommender.models.content_based import content_model
from book_recommender.models.hybrid_model import hybrid_model
from book_recommender.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


async def train_collaborative_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Train collaborative filtering model."""
    logger.info("Starting collaborative filtering model training")

    try:
        # Load data
        logger.info("Loading collaborative filtering data")
        books_df = data_loader.load_books_data()
        users_df = data_loader.load_users_data() 
        ratings_df = data_loader.load_ratings_data()

        # Validate data
        logger.info("Validating data")
        books_validation = data_validator.validate_books_dataframe(books_df)
        ratings_validation = data_validator.validate_ratings_dataframe(ratings_df)

        if not books_validation['valid'] or not ratings_validation['valid']:
            logger.error("Data validation failed")
            return {"status": "failed", "error": "Data validation failed"}

        # Clean data
        logger.info("Cleaning data")
        clean_books = data_preprocessor.clean_books_data(books_df)
        clean_ratings = data_preprocessor.clean_ratings_data(ratings_df)

        # Train model
        training_stats = collaborative_model.train(clean_ratings, clean_books)

        # Save model
        if not args.no_save:
            collaborative_model.save_model()
            logger.info("Collaborative filtering model saved")

        return {
            "status": "success",
            "model_type": "collaborative_filtering",
            "training_stats": training_stats
        }

    except Exception as e:
        logger.error(f"Collaborative filtering training failed: {e}")
        return {"status": "failed", "error": str(e)}


async def train_content_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Train content-based model."""
    logger.info("Starting content-based model training")

    try:
        # Load data
        logger.info("Loading content-based data")
        content_books_df = data_loader.load_content_books_data()

        # Validate data
        logger.info("Validating text data")
        texts = content_books_df['description'].fillna('').tolist()
        text_validation = data_validator.validate_text_data(texts)

        if not text_validation['valid']:
            logger.warning("Text validation issues found, proceeding with training")

        # Initialize embedding service
        logger.info("Initializing embedding service")
        corpus_texts = content_books_df.apply(
            lambda row: f"{row.get('title', '')} {row.get('description', '')} {row.get('categories', '')}",
            axis=1
        ).tolist()

        await embedding_service.initialize(corpus_texts)

        # Train model
        training_stats = await content_model.train(content_books_df, use_ollama=args.use_ollama)

        # Save model
        if not args.no_save:
            content_model.save_model()
            logger.info("Content-based model saved")

        return {
            "status": "success",
            "model_type": "content_based",
            "training_stats": training_stats
        }

    except Exception as e:
        logger.error(f"Content-based training failed: {e}")
        return {"status": "failed", "error": str(e)}


async def train_hybrid_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Train hybrid model."""
    logger.info("Starting hybrid model training")

    try:
        # Load all data
        logger.info("Loading all datasets for hybrid training")
        books_df = data_loader.load_books_data()
        ratings_df = data_loader.load_ratings_data()
        content_books_df = data_loader.load_content_books_data()

        # Clean data
        clean_books = data_preprocessor.clean_books_data(books_df)
        clean_ratings = data_preprocessor.clean_ratings_data(ratings_df)

        # Train hybrid model
        training_stats = await hybrid_model.train(
            clean_ratings,
            clean_books,
            content_books_df,
            use_ollama=args.use_ollama
        )

        # Save model
        if not args.no_save:
            hybrid_model.save_model()
            logger.info("Hybrid model saved")

        return {
            "status": "success",
            "model_type": "hybrid",
            "training_stats": training_stats
        }

    except Exception as e:
        logger.error(f"Hybrid training failed: {e}")
        return {"status": "failed", "error": str(e)}


def generate_training_report(results: Dict[str, Any]) -> None:
    """Generate training report."""
    logger.info("Generating training report")

    report_path = Path("training_report.md")

    with open(report_path, "w") as f:
        f.write("# Model Training Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for model_type, result in results.items():
            f.write(f"## {model_type.replace('_', ' ').title()}\n\n")

            if result["status"] == "success":
                f.write("✅ **Status:** Successfully trained\n\n")

                if "training_stats" in result:
                    f.write("### Training Statistics\n\n")
                    stats = result["training_stats"]

                    for key, value in stats.items():
                        f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                    f.write("\n")
            else:
                f.write("❌ **Status:** Training failed\n\n")
                f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")

        f.write("---\n\n")
        f.write("*Report generated by Book Recommender System training script*\n")

    logger.info(f"Training report saved to {report_path}")


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train book recommendation models")
    parser.add_argument(
        "--model",
        choices=["collaborative", "content", "hybrid", "all"],
        default="all",
        help="Model type to train"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Don't use Ollama for embeddings (use TF-IDF instead)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()
    args.use_ollama = not args.no_ollama

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("Starting model training pipeline")
    logger.info(f"Training model(s): {args.model}")
    logger.info(f"Use Ollama: {args.use_ollama}")
    logger.info(f"Save models: {not args.no_save}")

    # Training results
    results = {}

    start_time = time.time()

    try:
        if args.model in ["collaborative", "all"]:
            results["collaborative_filtering"] = await train_collaborative_model(args)

        if args.model in ["content", "all"]:
            results["content_based"] = await train_content_model(args)

        if args.model in ["hybrid", "all"]:
            results["hybrid"] = await train_hybrid_model(args)

        # Generate report
        generate_training_report(results)

        end_time = time.time()
        duration = end_time - start_time

        # Summary
        successful = sum(1 for r in results.values() if r["status"] == "success")
        total = len(results)

        logger.info(f"Training completed in {duration:.2f} seconds")
        logger.info(f"Successfully trained: {successful}/{total} models")

        if successful < total:
            logger.warning("Some models failed to train. Check the training report for details.")
            sys.exit(1)
        else:
            logger.info("All models trained successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if embedding_service:
            await embedding_service.close()


if __name__ == "__main__":
    asyncio.run(main())
