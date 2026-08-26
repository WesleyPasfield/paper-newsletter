from pathlib import Path
import re


TEMPLATE = Path(__file__).parents[1] / "template.yaml"


def test_ses_send_policies_select_the_deployed_identity_type():
    template = TEMPLATE.read_text()
    statements = re.findall(
        r"- Effect: Allow\n"
        r"\s+Action:\n"
        r"\s+- ses:Send(?:Bulk)?TemplatedEmail\n"
        r"(?P<resources>\s+Resource:\n(?:\s{16,}.*\n)+?)"
        r"(?=\s{12}- Effect:|\s{6}Events:)",
        template,
    )

    assert len(statements) == 4
    for resources in statements:
        assert "- !If" in resources
        assert "- CreateDomainIdentity" in resources
        assert "identity/${DomainName}" in resources
        assert "identity/${SenderEmail}" in resources
