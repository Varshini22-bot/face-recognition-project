"""Configurable recognition threshold decisions."""

from dataclasses import dataclass


DEFAULT_SIMILARITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class SimilarityThreshold:
	"""Apply one configurable cosine-similarity threshold.

	The default is an initial development value, not a scientifically calibrated
	production threshold. It should be tuned with a representative evaluation set.
	"""

	value: float = DEFAULT_SIMILARITY_THRESHOLD

	def __post_init__(self) -> None:
		if not -1.0 <= self.value <= 1.0:
			raise ValueError("similarity threshold must be between -1 and 1")

	def is_recognized(self, similarity: float) -> bool:
		"""Return whether a similarity meets the inclusive threshold."""
		return similarity >= self.value
