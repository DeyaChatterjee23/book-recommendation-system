#!/usr/bin/env python3
"""
Data pipeline script for the Book Recommender System.

This script handles:
1. Data download and validation
2. Data cleaning and preprocessing
3. Data splitting for training/testing
4. Data quality reports
"""

import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from book_recommender.core.logger import setup_logging
from book_recommender.data.data_loader import data_loader
from book_recommender.data.data_preprocessor import data_preprocessor
from book_recommender.data.data_validator import data_validator
from book_recommender.utils.helpers import ensure_directory

logger = logging.getLogger(__name__)


def download_sample_data():
    """Download sample data for demonstration purposes."""
    logger.info("Downloading sample data")

    # This is a placeholder - in production, you'd download from actual sources
    # For now, we'll create some sample data

    import pandas as pd
    import numpy as np

    raw_data_path = Path("data/raw")
    ensure_directory(raw_data_path)

    # Sample books data
    sample_books = pd.DataFrame({
        'ISBN': [f'ISBN{i:06d}' for i in range(1000)],
        'Book-Title': [f'Sample Book {i}' for i in range(1000)],
        'Book-Author': [f'Author {i % 100}' for i in range(1000)],
        'Year-Of-Publication': np.random.randint(1950, 2024, 1000),
        'Publisher': [f'Publisher {i % 50}' for i in range(1000)],
        'Image-URL-L': ['http://example.com/image.jpg'] * 1000
    })

    # Sample users data
    sample_users = pd.DataFrame({
        'User-ID': range(1, 501),
        'Location': [f'City {i % 100}, Country' for i in range(500)],
        'Age': np.random.randint(18, 80, 500)
    })

    # Sample ratings data
    np.random.seed(42)
    sample_ratings = []
    for user_id in range(1, 501):
        num_ratings = np.random.randint(10, 100)
        book_indices = np.random.choice(1000, num_ratings, replace=False)
        for book_idx in book_indices:
            rating = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                    p=[0.1, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])
            sample_ratings.append({
                'User-ID': user_id,
                'ISBN': f'ISBN{book_idx:06d}',
                'Book-Rating': rating
            })

    sample_ratings_df = pd.DataFrame(sample_ratings)

    # Save sample data
    sample_books.to_csv(raw_data_path / "BX-Books.csv", sep=';', index=False)
    sample_users.to_csv(raw_data_path / "BX-Users.csv", sep=';', index=False)
    sample_ratings_df.to_csv(raw_data_path / "BX-Book-Ratings.csv", sep=';', index=False)

    # Sample content-based data
    categories = ['Fiction', 'Non-fiction', 'Science', 'History', 'Biography', 'Romance', 'Mystery', 'Fantasy']
    sample_content_books = pd.DataFrame({
        'title': [f'Content Book {i}' for i in range(1000)],
        'authors': [f'Author {i % 100}' for i in range(1000)],
        'description': [f'This is a sample description for book {i}. ' * 10 for i in range(1000)],
        'categories': [np.random.choice(categories) for _ in range(1000)],
        'published_year': np.random.randint(1950, 2024, 1000),
        'average_rating': np.random.uniform(2.0, 5.0, 1000),
        'num_pages': np.random.randint(100, 1000, 1000),
        'ratings_count': np.random.randint(10, 10000, 1000)
    })

    sample_content_books.to_csv(raw_data_path / "7kbooks.csv", index=False)

    logger.info("Sample data created successfully")


def process_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Process and validate data."""
    logger.info("Starting data processing pipeline")

    results = {
        "books": {"status": "not_processed"},
        "users": {"status": "not_processed"},
        "ratings": {"status": "not_processed"},
        "content_books": {"status": "not_processed"}
    }

    try:
        # Load raw data
        logger.info("Loading raw data")
        books_df = data_loader.load_books_data()
        users_df = data_loader.load_users_data()
        ratings_df = data_loader.load_ratings_data()
        content_books_df = data_loader.load_content_books_data()

        # Validate raw data
        logger.info("Validating raw data")
        books_validation = data_validator.validate_books_dataframe(books_df)
        ratings_validation = data_validator.validate_ratings_dataframe(ratings_df)

        results["books"]["validation"] = books_validation
        results["ratings"]["validation"] = ratings_validation

        if not books_validation["valid"]:
            logger.error("Books data validation failed")
            results["books"]["status"] = "validation_failed"
        else:
            results["books"]["status"] = "validated"

        if not ratings_validation["valid"]:
            logger.error("Ratings data validation failed")
            results["ratings"]["status"] = "validation_failed"
        else:
            results["ratings"]["status"] = "validated"

        # Clean data
        logger.info("Cleaning data")
        clean_books = data_preprocessor.clean_books_data(books_df)
        clean_ratings = data_preprocessor.clean_ratings_data(ratings_df)
        clean_content_books = data_preprocessor.prepare_content_features(content_books_df)

        # Generate data quality reports
        logger.info("Generating data quality reports")
        books_quality_report = data_validator.generate_data_quality_report(clean_books, "books")
        ratings_quality_report = data_validator.generate_data_quality_report(clean_ratings, "ratings")
        content_quality_report = data_validator.generate_data_quality_report(clean_content_books, "content_books")

        results["books"]["quality_report"] = books_quality_report
        results["ratings"]["quality_report"] = ratings_quality_report
        results["content_books"]["quality_report"] = content_quality_report

        # Save processed data
        if not args.no_save:
            logger.info("Saving processed data")
            data_loader.save_processed_data(clean_books, "clean_books.csv")
            data_loader.save_processed_data(clean_ratings, "clean_ratings.csv")
            data_loader.save_processed_data(clean_content_books, "clean_content_books.csv")

        # Split data for training/testing
        if args.split_data:
            logger.info("Splitting data for training and testing")
            train_ratings, test_ratings = data_preprocessor.split_data(clean_ratings)

            if not args.no_save:
                data_loader.save_processed_data(train_ratings, "train_ratings.csv")
                data_loader.save_processed_data(test_ratings, "test_ratings.csv")

            results["data_split"] = {
                "train_size": len(train_ratings),
                "test_size": len(test_ratings),
                "split_ratio": len(train_ratings) / len(clean_ratings)
            }

        # Update status for successful processing
        results["books"]["status"] = "processed"
        results["users"]["status"] = "processed" 
        results["ratings"]["status"] = "processed"
        results["content_books"]["status"] = "processed"

        return results

    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return {"error": str(e), "status": "failed"}


def generate_data_report(results: Dict[str, Any]) -> None:
    """Generate data processing report."""
    logger.info("Generating data processing report")

    report_path = Path("data_processing_report.md")

    with open(report_path, "w") as f:
        f.write("# Data Processing Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if "error" in results:
            f.write("❌ **Status:** Data processing failed\n\n")
            f.write(f"**Error:** {results['error']}\n\n")
            return

        f.write("## Data Processing Summary\n\n")

        for dataset, result in results.items():
            if dataset == "data_split":
                continue

            f.write(f"### {dataset.title()}\n\n")

            status = result.get("status", "unknown")
            if status == "processed":
                f.write("✅ **Status:** Successfully processed\n\n")
            elif status == "validated":
                f.write("⚠️ **Status:** Validated but not processed\n\n")
            elif status == "validation_failed":
                f.write("❌ **Status:** Validation failed\n\n")
            else:
                f.write(f"❓ **Status:** {status}\n\n")

            # Quality report
            if "quality_report" in result:
                quality = result["quality_report"]
                f.write(f"**Data Quality Score:** {quality['data_quality_score']:.2f}/1.0\n\n")
                f.write(f"**Total Records:** {quality['basic_info']['total_records']:,}\n")
                f.write(f"**Memory Usage:** {quality['basic_info']['memory_usage_mb']:.2f} MB\n")
                f.write(f"**Duplicate Records:** {quality['basic_info']['duplicate_records']:,}\n\n")

                if quality.get('recommendations'):
                    f.write("**Recommendations:**\n")
                    for rec in quality['recommendations']:
                        f.write(f"- {rec}\n")
                    f.write("\n")

        # Data split info
        if "data_split" in results:
            f.write("## Data Split\n\n")
            split_info = results["data_split"]
            f.write(f"- **Training Set:** {split_info['train_size']:,} records\n")
            f.write(f"- **Test Set:** {split_info['test_size']:,} records\n")
            f.write(f"- **Split Ratio:** {split_info['split_ratio']:.2%}\n\n")

        f.write("---\n\n")
        f.write("*Report generated by Book Recommender System data pipeline*\n")

    logger.info(f"Data processing report saved to {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process data for book recommendation system")
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Download sample data for testing"
    )
    parser.add_argument(
        "--split-data",
        action="store_true",
        help="Split data into training and test sets"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save processed data"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    logger.info("Starting data pipeline")

    start_time = time.time()

    try:
        # Download sample data if requested
        if args.download_sample:
            download_sample_data()

        # Process data
        results = process_data(args)

        # Generate report
        generate_data_report(results)

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Data pipeline completed in {duration:.2f} seconds")

        if "error" in results:
            logger.error("Data pipeline failed")
            sys.exit(1)
        else:
            logger.info("Data pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Data pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
