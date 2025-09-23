'''
We will implement gradient descent from scratch using numpy, pandas, and matplotlib.

This small helper ensures the Kaggle dataset identifier is sanitized before
calling the kaggle API and provides clearer error messages when something goes wrong.
'''
import sys
import kaggle


def sanitize_dataset_identifier(dataset_id: str) -> str:
	"""Normalize a dataset identifier and remove common trailing segments like '/data'.

	The kaggle API expects identifiers of the form 'owner/dataset-name'. Sometimes
	users pass a path like 'owner/dataset-name/data' which can confuse the SDK.
	"""
	if not isinstance(dataset_id, str):
		raise TypeError("dataset identifier must be a string")
	dataset_id = dataset_id.strip()
	parts = dataset_id.split('/')
	# If the last segment is 'data' (a common mistake), drop it
	if len(parts) > 2 and parts[-1].lower() == 'data':
		return '/'.join(parts[:-1])
	return dataset_id


def main() -> None:
	# original value the script used; keep it here for backwards compatibility
	raw_dataset = 'nabihazahid/spotify-dataset-for-churn-analysis/data'
	dataset = sanitize_dataset_identifier(raw_dataset)
	try:
		print(f"Downloading dataset '{dataset}' to './' and unzipping...")
		kaggle.api.dataset_download_files(dataset, path='./', unzip=True)
		print('Download complete.')
	except TypeError as e:
		# Surface helpful troubleshooting info for the user
		print('TypeError from kaggle API:', e)
		print("This often means an incorrect parameter (for example dataset_version_number)")
		print("was passed to the Kaggle SDK. Ensure the dataset identifier is 'owner/dataset'.")
		print("If you need to specify a version, pass an integer to dataset_version_number.")
		sys.exit(1)
	except Exception as e:
		print('Failed to download dataset:', e)
		print('Ensure you have a valid kaggle.json credentials file and the dataset id is correct.')
		sys.exit(1)


if __name__ == '__main__':
	main()

