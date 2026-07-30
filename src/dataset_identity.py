from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from output_paths import slugify_component


IDRID_TYPES = frozenset({"idrid"})
UWF_TYPES = frozenset({"ch", "co", "nm", "rb", "rch", "um"})
FASHIONIQ_TYPES = frozenset({"dress", "shirt", "toptee"})

ROOT_NAME_TO_DATASET: Dict[str, str] = {
    "idrid_cir_dataset_cold": "idrid",
    "uwf_cir_dataset_cold": "uwf",
    "combined_fundus_cir_dataset": "combined-fundus-cir",
    "fashioniq_dataset": "fashioniq",
}


class DatasetIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_slug: str
    dataset_format: str
    root_requested: Optional[str]
    root_resolved: Optional[str]
    classification_evidence: Tuple[str, ...]

    def layout_fields(self) -> dict:
        return {
            "dataset": self.dataset_slug,
            "dataset_format": self.dataset_format,
            "dataset_root_requested": self.root_requested,
            "dataset_root_resolved": self.root_resolved,
            "dataset_classification_evidence": self.classification_evidence,
        }


def _classify_categories(category_set: frozenset) -> Optional[str]:
    if category_set == IDRID_TYPES:
        return "idrid"
    if category_set == UWF_TYPES:
        return "uwf"
    if category_set == frozenset({"internal"}):
        return "combined-fundus-cir"
    if category_set == FASHIONIQ_TYPES:
        return "fashioniq"
    return None


def _classify_root(resolved_root: Optional[str]) -> Optional[str]:
    if resolved_root is None:
        return None
    return ROOT_NAME_TO_DATASET.get(Path(resolved_root).name.casefold())


def resolve_dataset_identity(
    *,
    dataset_format: str,
    dress_types: Sequence[str],
    dataset_root_requested: Optional[Union[str, Path]],
    dataset_root_resolved: Optional[Union[str, Path]],
    output_dataset: Optional[str],
) -> DatasetIdentity:
    normalized_format = slugify_component(dataset_format)
    requested = (
        str(Path(dataset_root_requested).expanduser())
        if dataset_root_requested is not None
        else None
    )
    resolved = (
        str(Path(dataset_root_resolved).expanduser().resolve())
        if dataset_root_resolved is not None
        else None
    )

    if normalized_format == "cirr":
        return DatasetIdentity(
            dataset_slug="cirr",
            dataset_format="cirr",
            root_requested=requested,
            root_resolved=resolved,
            classification_evidence=("dataset-format:cirr",),
        )

    if output_dataset is not None:
        return DatasetIdentity(
            dataset_slug=slugify_component(output_dataset),
            dataset_format=normalized_format,
            root_requested=requested,
            root_resolved=resolved,
            classification_evidence=("explicit-output-dataset",),
        )

    category_set = frozenset(value.casefold() for value in dress_types)
    category_slug = _classify_categories(category_set)
    root_slug = _classify_root(resolved)
    evidence = tuple(
        value
        for value in (
            f"resolved-root:{resolved}" if root_slug else None,
            (
                f"dress-types:{','.join(sorted(category_set))}"
                if category_slug
                else None
            ),
        )
        if value is not None
    )
    candidates = {value for value in (root_slug, category_slug) if value}
    if len(candidates) > 1:
        raise DatasetIdentityError(
            f"conflicting dataset evidence: root={root_slug}, "
            f"dress_types={category_slug}"
        )
    if not candidates:
        raise DatasetIdentityError(
            "could not identify actual dataset from root or dress types"
        )
    return DatasetIdentity(
        dataset_slug=candidates.pop(),
        dataset_format=normalized_format,
        root_requested=requested,
        root_resolved=resolved,
        classification_evidence=evidence,
    )


def resolve_fashioniq_training_identity(
    *,
    project_root: Path,
    dress_types: Sequence[str],
    dataset_root: Optional[Union[str, Path]],
    output_dataset: Optional[str],
    root_resolver: Callable[
        [str, str, Optional[Union[str, Path]]], Path
    ],
) -> DatasetIdentity:
    resolved_roots = {
        root_resolver(category, "train", dataset_root).resolve()
        for category in dress_types
    }
    if len(resolved_roots) != 1:
        raise DatasetIdentityError(
            "FashionIQ categories resolved to multiple dataset roots: "
            + ", ".join(sorted(str(path) for path in resolved_roots))
        )

    resolved_root = resolved_roots.pop()
    requested_root = dataset_root
    project_link = project_root / "fashionIQ_dataset"
    if requested_root is None and (
        project_link.exists() or project_link.is_symlink()
    ):
        if project_link.resolve() == resolved_root:
            requested_root = str(project_link)

    return resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=dress_types,
        dataset_root_requested=requested_root,
        dataset_root_resolved=resolved_root,
        output_dataset=output_dataset,
    )
