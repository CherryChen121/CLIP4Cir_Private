from pathlib import Path

import pytest

from dataset_identity import (
    DatasetIdentityError,
    resolve_dataset_identity,
    resolve_fashioniq_evaluation_identity,
    resolve_fashioniq_training_identity,
)


@pytest.mark.parametrize(
    ("root_name", "dress_types", "expected"),
    [
        ("IDRiD_CIR_Dataset_cold", ["IDRiD"], "idrid"),
        (
            "UWF_CIR_Dataset_cold",
            ["CH", "CO", "NM", "RB", "RCH", "UM"],
            "uwf",
        ),
        (
            "Combined_Fundus_CIR_Dataset",
            ["Internal"],
            "combined-fundus-cir",
        ),
        ("fashionIQ_dataset", ["dress", "shirt", "toptee"], "fashioniq"),
    ],
)
def test_resolves_supported_fashioniq_format_datasets(
    tmp_path, root_name, dress_types, expected
):
    root = tmp_path / root_name
    root.mkdir()

    identity = resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=dress_types,
        dataset_root_requested=str(root),
        dataset_root_resolved=root,
        output_dataset=None,
    )

    assert identity.dataset_slug == expected
    assert identity.dataset_format == "fashioniq"
    assert identity.root_requested == str(root)
    assert identity.root_resolved == str(root.resolve())


def test_explicit_output_dataset_wins_and_is_slugified(tmp_path):
    root = tmp_path / "unrecognized"
    root.mkdir()

    identity = resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=["unknown"],
        dataset_root_requested="unrecognized",
        dataset_root_resolved=root,
        output_dataset=" My Study / Phase 1 ",
    )

    assert identity.dataset_slug == "my-study-phase-1"
    assert "explicit-output-dataset" in identity.classification_evidence


def test_conflicting_root_and_categories_are_rejected(tmp_path):
    root = tmp_path / "IDRiD_CIR_Dataset_cold"
    root.mkdir()

    with pytest.raises(DatasetIdentityError, match="conflicting"):
        resolve_dataset_identity(
            dataset_format="fashioniq",
            dress_types=["CH", "CO", "NM", "RB", "RCH", "UM"],
            dataset_root_requested=str(root),
            dataset_root_resolved=root,
            output_dataset=None,
        )


def test_unknown_automatic_evidence_is_rejected(tmp_path):
    root = tmp_path / "mystery"
    root.mkdir()

    with pytest.raises(DatasetIdentityError, match="could not identify"):
        resolve_dataset_identity(
            dataset_format="fashioniq",
            dress_types=["unknown"],
            dataset_root_requested=None,
            dataset_root_resolved=root,
            output_dataset=None,
        )


def test_cirr_identity_does_not_require_fashioniq_evidence():
    identity = resolve_dataset_identity(
        dataset_format="cirr",
        dress_types=(),
        dataset_root_requested=None,
        dataset_root_resolved=None,
        output_dataset=None,
    )

    assert identity.dataset_slug == "cirr"
    assert identity.dataset_format == "cirr"


def test_training_identity_records_requested_symlink_and_resolved_target(
    tmp_path,
):
    target = tmp_path / "IDRiD_CIR_Dataset_cold"
    target.mkdir()
    link = tmp_path / "fashionIQ_dataset"
    link.symlink_to(target, target_is_directory=True)

    identity = resolve_fashioniq_training_identity(
        project_root=tmp_path,
        dress_types=["IDRiD"],
        dataset_root=None,
        output_dataset=None,
        root_resolver=lambda *_args: link,
    )

    assert identity.dataset_slug == "idrid"
    assert identity.root_requested == str(link)
    assert identity.root_resolved == str(target.resolve())


def test_training_identity_rejects_categories_resolved_to_multiple_roots(
    tmp_path,
):
    roots = {
        "CH": tmp_path / "UWF_CIR_Dataset_cold",
        "CO": tmp_path / "other-UWF-root",
    }
    for root in roots.values():
        root.mkdir()

    with pytest.raises(DatasetIdentityError, match="multiple dataset roots"):
        resolve_fashioniq_training_identity(
            project_root=tmp_path,
            dress_types=["CH", "CO"],
            dataset_root=None,
            output_dataset=None,
            root_resolver=lambda category, *_args: roots[category],
        )


def test_evaluation_identity_uses_requested_root_and_split(tmp_path):
    root = tmp_path / "Combined_Fundus_CIR_Dataset"
    root.mkdir()
    calls = []

    identity = resolve_fashioniq_evaluation_identity(
        project_root=tmp_path,
        dress_types=["ODIR5K", "GRAPE"],
        split="test",
        dataset_root=str(root),
        output_dataset="combined-fundus-cir",
        root_resolver=lambda category, split, requested: (
            calls.append((category, split, requested)) or root
        ),
    )

    assert calls == [
        ("ODIR5K", "test", str(root)),
        ("GRAPE", "test", str(root)),
    ]
    assert identity.dataset_slug == "combined-fundus-cir"
    assert identity.root_requested == str(root)
    assert identity.root_resolved == str(root.resolve())


def test_evaluation_identity_classifies_actual_root_without_override(
    tmp_path,
):
    root = tmp_path / "Combined_Fundus_CIR_Dataset"
    root.mkdir()

    identity = resolve_fashioniq_evaluation_identity(
        project_root=tmp_path,
        dress_types=["ODIR5K"],
        split="test",
        dataset_root=root,
        output_dataset=None,
        root_resolver=lambda *_args: root,
    )

    assert identity.dataset_slug == "combined-fundus-cir"
    assert identity.dataset_format == "fashioniq"


def test_evaluation_identity_rejects_categories_on_multiple_roots(
    tmp_path,
):
    first = tmp_path / "IDRiD_CIR_Dataset_cold"
    second = tmp_path / "UWF_CIR_Dataset_cold"
    first.mkdir()
    second.mkdir()

    with pytest.raises(DatasetIdentityError, match="multiple dataset roots"):
        resolve_fashioniq_evaluation_identity(
            project_root=tmp_path,
            dress_types=["IDRiD", "CH"],
            split="val",
            dataset_root=None,
            output_dataset=None,
            root_resolver=lambda category, *_args: (
                first if category == "IDRiD" else second
            ),
        )


def test_layout_fields_match_create_run_layout_interface(tmp_path):
    root = tmp_path / "IDRiD_CIR_Dataset_cold"
    root.mkdir()
    identity = resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=["IDRiD"],
        dataset_root_requested=root,
        dataset_root_resolved=root,
        output_dataset=None,
    )

    assert identity.layout_fields() == {
        "dataset": "idrid",
        "dataset_format": "fashioniq",
        "dataset_root_requested": str(root),
        "dataset_root_resolved": str(root.resolve()),
        "dataset_classification_evidence": (
            f"resolved-root:{root.resolve()}",
            "dress-types:idrid",
        ),
    }
