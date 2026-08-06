from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_skill_package_exists() -> None:
    assert (ROOT / "SKILL.md").is_file()
    assert (ROOT / "agents" / "openai.yaml").is_file()
    for reference in (
        "agent-contract.md",
        "model-and-output-contract.md",
        "data-and-causality-contract.md",
        "evaluation-and-evidence-contract.md",
    ):
        assert (ROOT / "references" / reference).is_file()


def test_skill_frontmatter_and_progressive_references() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    frontmatter_text = skill_text.split("---", 2)[1].strip()
    frontmatter = dict(
        line.split(":", 1) for line in frontmatter_text.splitlines() if ":" in line
    )
    frontmatter = {key.strip(): value.strip() for key, value in frontmatter.items()}

    assert set(frontmatter) == {"name", "description"}
    assert frontmatter["name"] == "skill-dl-tcn-shortterm"
    for trigger in (
        "Temporal Convolutional Network",
        "横截面收益排序",
        "PIT/walk-forward/purge/embargo",
        "RankIC",
        "训练吞吐",
    ):
        assert trigger in frontmatter["description"]

    for reference in (
        "references/agent-contract.md",
        "references/model-and-output-contract.md",
        "references/data-and-causality-contract.md",
        "references/evaluation-and-evidence-contract.md",
    ):
        assert reference in skill_text


def test_skill_ui_metadata_matches_skill_name() -> None:
    metadata = (ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert "$skill-dl-tcn-shortterm" in metadata
    short_line = next(
        line for line in metadata.splitlines() if "short_description:" in line
    )
    short_description = short_line.split(":", 1)[1].strip().strip('"')
    assert 25 <= len(short_description) <= 64


def test_skill_preserves_model_strategy_and_release_boundaries() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    evidence = (ROOT / "references" / "evaluation-and-evidence-contract.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "TCN 不决定持仓、TopK、换手缓冲或成交规则",
        "control_tcn",
        "V42 consensus student",
        "不得外推为跨硬件、跨数据的普遍结论",
    ):
        assert required in skill_text

    for required in (
        "v46_student_not_generalized",
        "alpha_ready=false",
        "deployment_authorized=false",
        "trading_authorized=false",
    ):
        assert required in evidence


def test_skill_contains_no_machine_specific_or_secret_material() -> None:
    package_files = [
        ROOT / "README.md",
        ROOT / "SKILL.md",
        ROOT / "agents" / "openai.yaml",
        *sorted((ROOT / "references").glob("*.md")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in package_files)
    forbidden = (
        "D:\\codexapp",
        "/home/a666",
        "api_key",
        "password=",
    )
    for value in forbidden:
        assert value not in text


def test_skill_exposes_one_flat_agent_interface() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    for command in (
        "tcn-shortterm-skill demo",
        "tcn-shortterm-skill run --request <request.json>",
        "tcn-shortterm-skill example --output-dir <empty-directory>",
        "tcn-shortterm-skill schema --kind request",
        "tcn-shortterm-skill schema --kind result",
    ):
        assert command in skill_text

    assert "python tasks/" not in skill_text
    assert "仓库根目录执行" not in skill_text
