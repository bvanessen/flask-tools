# server.py
# pip install "mcp[cli]" rdkit-pypi
import click
from typing_extensions import TypedDict
from loguru import logger
from typing import Literal, List
from fastmcp import FastMCP

# import your existing module (the one we’ve been building)
import flask_tools.chemistry.polymerizer as pr


# ----- Structured outputs for nicer tool schemas -----
class Suggestion(TypedDict):
    strategy: str
    confidence: float
    reason: str


class PolymerizeResult(TypedDict):
    repeat_smiles: str
    strategy: str
    rationale: str


# ----- Tools -----

from flask_tools.utils.server_utils import get_hostname
from lc_conductor.tool_registration import register_tool_server


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    help="MCP transport type",
    default="streamable-http",
)
@click.option("--port", type=int, default=8129, help="Port to run the server on")
@click.option("--host", type=str, default=None, help="Host to run the server on")
@click.option(
    "--name", type=str, default="polymerizer_tools", help="Name of the MCP server"
)
@click.option(
    "--copilot-port", type=int, default=8001, help="Port to the running copilot backend"
)
@click.option(
    "--copilot-host", type=str, default=None, help="Host to the running copilot backend"
)
def main(
    transport: str,
    port: str,
    host: str,
    name: str,
    copilot_port: str,
    copilot_host: str,
):
    if host is None:
        _, host = get_hostname()

    # Init MCP server
    mcp = FastMCP(
        "Polymerizer",
        # description=(
        # "Expose monomer→polymer repeat transforms via MCP tools. "
        # "Tools: polymerize_explicit, polymerize_auto, suggest_rules."
    )

    @mcp.tool()
    def polymerize_explicit(
        monomer_smiles: str,
        strategy: Literal[
            "vinyl",
            "acrylate",
            "rop_thf",
            "rop_epoxide",
            "ketene",
            "cond_alpha_hydroxy_acid",
            "cond_diphenol",
            "rop_lactam",
            "cond_omega_amino_acid",
            "alkyne",
            "polyacetylene",  # optional pretty-printer for C#C
        ],
        bigsmiles_wrap: bool = False,
    ) -> str:
        """
        Apply a specific polymerization rule and return the repeat-unit SMILES with [*] endpoints.
        Set bigsmiles_wrap=True to wrap in a simple {…} block.
        """
        rep = pr.monomer_to_repeat_smiles(monomer_smiles, strategy=strategy)
        return pr.wrap_bigsmiles_like(rep) if bigsmiles_wrap else rep

    @mcp.tool()
    def suggest_rules(monomer_smiles: str, top_k: int = 5) -> List[Suggestion]:
        """
        Inspect a monomer and return ranked candidate strategies with reasons.
        This does NOT perform any transformation.
        """
        ranked = pr.suggest_polymerization_rules(monomer_smiles)
        out = [
            {
                "strategy": s.strategy,
                "confidence": float(s.confidence),
                "reason": s.reason,
            }
            for s in ranked[:top_k]
        ]
        return out

    @mcp.tool()
    def polymerize_auto(
        monomer_smiles: str,
        min_confidence: float = 0.80,
        allow_fallback_to_lower_confidence: bool = True,
        bigsmiles_wrap: bool = False,
    ) -> PolymerizeResult:
        """
        Auto-select and apply a single-monomer rule.
        Returns repeat SMILES, chosen strategy, and rationale.
        Raises a helpful error if a comonomer is required or the case is ambiguous.
        """
        rep, strat, why = pr.monomer_to_repeat_auto(
            monomer_smiles,
            min_confidence=min_confidence,
            allow_fallback_to_lower_confidence=allow_fallback_to_lower_confidence,
        )
        rep_out = pr.wrap_bigsmiles_like(rep) if bigsmiles_wrap else rep
        return {"repeat_smiles": rep_out, "strategy": strat, "rationale": why}

    try:
        register_tool_server(port, host, name, copilot_port, copilot_host)
    except:
        logger.info(
            f"{name} could not connect to server for registration -- requires manual registration"
        )

    # Run MCP server
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=f"/{name}/mcp",
        json_response=True,
    )


# ----- Run the server -----
if __name__ == "__main__":
    main()
