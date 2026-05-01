"""Disease ID cross-references: OMIM / Orphanet → Mondo (and reverse)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DiseaseIdMapper:
    def __init__(
        self,
        omim2mondo_path: Optional[str] = None,
        orphanet2mondo_path: Optional[str] = None,
    ):
        self.omim2mondo: Dict[str, str] = {}
        self.orphanet2mondo: Dict[str, str] = {}
        self.mondo2omim: Dict[str, List[str]] = {}
        self.mondo2orphanet: Dict[str, List[str]] = {}

        if omim2mondo_path:
            self._load(omim2mondo_path, "omim")
        if orphanet2mondo_path:
            self._load(orphanet2mondo_path, "orphanet")

    def _load(self, path: str, kind: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Disease ID map not found: {path}")
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if kind == "omim":
            target = self.omim2mondo
            reverse = self.mondo2omim
        else:
            target = self.orphanet2mondo
            reverse = self.mondo2orphanet
        for k, v in (data or {}).items():
            if isinstance(v, list):
                if v:
                    target[k] = v[0]
                    for mondo in v:
                        reverse.setdefault(mondo, []).append(k)
            else:
                target[k] = v
                reverse.setdefault(v, []).append(k)
        logger.info(f"Loaded {len(target)} {kind} → Mondo mappings")

    def omim_to_mondo(self, omim: str) -> Optional[str]:
        return self.omim2mondo.get(omim)

    def orphanet_to_mondo(self, orpha: str) -> Optional[str]:
        return self.orphanet2mondo.get(orpha)

    def mondo_to_omim(self, mondo: str) -> List[str]:
        return self.mondo2omim.get(mondo, [])

    def mondo_to_orphanet(self, mondo: str) -> List[str]:
        return self.mondo2orphanet.get(mondo, [])
    def to_mondo(self, disease_id: str) -> Optional[str]:
            """Prefix-dispatching conversion to Mondo ID.

            Handles:
            - "MONDO:..." (pass-through)
            - "OMIM:...", "OMIM..." (look up via omim2mondo)
            - "ORPHA:...", "Orphanet:..." (look up via orphanet2mondo)
            Returns None if no mapping is known.
            """
            if not disease_id:
                return None
            did = str(disease_id).strip()

            if did.upper().startswith("MONDO"):
                return did

            if did.upper().startswith("OMIM"):
                return self.omim2mondo.get(did) or self.omim2mondo.get(did.replace("OMIM:", ""))

            if did.upper().startswith("ORPHA") or did.upper().startswith("ORPHANET"):
                # Normalize variants: "ORPHA:123", "Orphanet:123", "ORPHA123" → try all
                keys_to_try = [did]
                for prefix in ("ORPHA:", "Orphanet:", "ORPHA"):
                    if did.startswith(prefix):
                        suffix = did[len(prefix):]
                        keys_to_try.extend([
                            f"ORPHA:{suffix}", f"Orphanet:{suffix}",
                            f"ORPHA{suffix}", suffix,
                        ])
                for k in keys_to_try:
                    if k in self.orphanet2mondo:
                        return self.orphanet2mondo[k]
                return None

            return None