"""Chemical and aerosol mapping presets for CAMS, MERRA-2, and EMAC -> WRF-Chem."""
from __future__ import annotations

from airblender.physics.dsl import IcBcMappingItem


def get_cams2wrf_mapping(scheme: str = "made_sorgam") -> list[IcBcMappingItem]:
    """
    CAMS -> WRF-Chem (`chem_opt = 100/106`, RACM/RADM2 + MADE/SORGAM + GOCART) mapping.
    Exact operational multipliers for `ghg` (`0.1 deg` CO2/CH4) and `ml` streams.
    """
    scheme_key = str(scheme).lower()
    if scheme_key not in ("made_sorgam", "radm2_made", "racm_made", "default"):
        raise ValueError(f"Unsupported CAMS mapping scheme '{scheme}'; expected 'made_sorgam'")

    ml_stream = "ml"
    ghg_stream = "ghg"
    return [
        # --- Greenhouse Gases (High-res 0.1 deg CAMS GHG stream; falls back to ml for ch4) ---
        IcBcMappingItem("co2 -> 0.6582*[co2];1.e6", ghg_stream),
        IcBcMappingItem("ch4 -> 1.8058*[ch4];1.e6", ghg_stream),
        # --- Inorganics (Gases) ---
        IcBcMappingItem("o3 -> 0.604*[go3];1.e6", ml_stream),
        IcBcMappingItem("co -> 1.034*[co];1.e6", ml_stream),
        IcBcMappingItem("no -> 0.965*[no];1.e6", ml_stream),
        IcBcMappingItem("no2 -> 0.630*[no2];1.e6", ml_stream),
        IcBcMappingItem("so2 -> 0.452*[so2];1.e6", ml_stream),
        IcBcMappingItem("hno3 -> 0.460*[hno3];1.e6", ml_stream),
        IcBcMappingItem("h2o2 -> 0.852*[h2o2];1.e6", ml_stream),
        IcBcMappingItem("nh3 -> 1.701*[nh3];1.e6", ml_stream),
        IcBcMappingItem("onit -> 0.243*[onit];1.e6", ml_stream),
        # --- Nighttime / Reservoir Species ---
        IcBcMappingItem("n2o5 -> 0.268*[n2o5];1.e6", ml_stream),
        IcBcMappingItem("hno4 -> 0.367*[ho2no2];1.e6", ml_stream),
        IcBcMappingItem("pan -> 0.239*[pan];1.e6", ml_stream),
        IcBcMappingItem("no3 -> 0.467*[no3];1.e6", ml_stream),
        IcBcMappingItem("ho -> 1.703*[oh];1.e6", ml_stream),
        IcBcMappingItem("ho2 -> 0.878*[ho2];1.e6", ml_stream),
        # --- VOCs (Lumped RADM2 / RACM) ---
        IcBcMappingItem("hcho -> 0.965*[hcho];1.e6", ml_stream),
        IcBcMappingItem("ald -> 0.658*[ald2];1.e6", ml_stream),
        IcBcMappingItem("iso -> 0.425*[c5h8];1.e6", ml_stream),
        IcBcMappingItem("eth -> 0.963*[c2h6];1.e6", ml_stream),
        IcBcMappingItem("hc3 -> 0.657*[c3h8];1.e6", ml_stream),
        IcBcMappingItem("olt -> 0.688*[c3h6]+0.965*[ole];1.e6", ml_stream),
        IcBcMappingItem("ket -> 0.499*[ch3coch3]+0.628*[c2h5oh];1.e6", ml_stream),
        IcBcMappingItem("ora2 -> 0.629*[hcooh]+0.904*[ch3oh]+0.482*[mcooh];1.e6", ml_stream),
        IcBcMappingItem("hc5 -> 1.0*[par];1.e6", ml_stream),
        IcBcMappingItem("mgly -> 0.402*[ch3cocho];1.e6", ml_stream),
        IcBcMappingItem("api -> 0.213*[c10h16];1.e6", ml_stream),
        # --- Aerosols (GOCART Bins -> MADE/SORGAM+GOCART) ---
        IcBcMappingItem("ac0 -> 3.56*[aermr11];1.e11", ml_stream),
        IcBcMappingItem("so4aj -> 1.0*[aermr11];1.e9", ml_stream),
        IcBcMappingItem("corn -> 1.68*[aermr04]+1.68*[aermr05]+1.68*[aermr06];1.e7", ml_stream),
        IcBcMappingItem("soila -> 1.0*[aermr04]+1.0*[aermr05]+1.0*[aermr06];1.e9", ml_stream),
        IcBcMappingItem("corn -> 1.98*[aermr01]+1.98*[aermr02]+1.98*[aermr03];1.e7", ml_stream),
        IcBcMappingItem("seas -> 1.0*[aermr01]+1.0*[aermr02]+1.0*[aermr03];1.e9", ml_stream),
        IcBcMappingItem("ac0 -> 6.41*[aermr09]+6.41*[aermr10];1.e11", ml_stream),
        IcBcMappingItem("ecj -> 1.0*[aermr09]+1.0*[aermr10];1.e9", ml_stream),
        IcBcMappingItem("ac0 -> 6.41*[aermr07]+6.41*[aermr08];1.e11", ml_stream),
        IcBcMappingItem("orgpaj -> 1.0*[aermr07]+1.0*[aermr08];1.e9", ml_stream),
        IcBcMappingItem("ac0 -> 3.7*[nh4];1.e11", ml_stream),
        IcBcMappingItem("nh4aj -> 1.0*[nh4];1.e9", ml_stream),
        IcBcMappingItem("ac0 -> 4.24*[no3_a];1.e11", ml_stream),
        IcBcMappingItem("no3aj -> 1.0*[no3_a];1.e9", ml_stream),
    ]


def get_merra2wrf_mapping(scheme: str = "made_sorgam") -> list[IcBcMappingItem]:
    """
    MERRA-2 -> WRF-Chem mapping across supported target aerosol schemes:
    `made_sorgam` (default, chem_opt 100/106), `gocart`, `gocart_dust`, or `mosaic_8bin`.
    """
    gas_stream = "inst3_3d_chm_Nv"
    aer_stream = "inst3_3d_aer_Nv"
    scheme_key = str(scheme).lower()

    if scheme_key in ("made_sorgam", "radm2_made", "racm_made", "default"):
        return [
            IcBcMappingItem("so2 -> 0.453*[SO2];1.e6", aer_stream),
            IcBcMappingItem("o3 -> 0.604*[O3];1.e6", gas_stream),
            IcBcMappingItem("co -> 0.966*[CO];1.e6", gas_stream),
            IcBcMappingItem("ac0 -> 3.56*[SO4];1.e11", aer_stream),
            IcBcMappingItem("so4aj -> 1.0*[SO4];1.e9", aer_stream),
            IcBcMappingItem(
                "corn -> 1.68*[DU001]+1.68*[DU002]+1.68*[DU003]+1.68*[DU004]+1.68*[DU005];1.e7",
                aer_stream,
            ),
            IcBcMappingItem(
                "soila -> 1.0*[DU001]+1.0*[DU002]+1.0*[DU003]+1.0*[DU004]+1.0*[DU005];1.e9",
                aer_stream,
            ),
            IcBcMappingItem(
                "corn -> 1.98*[SS001]+1.98*[SS002]+1.98*[SS003]+1.98*[SS004]+1.98*[SS005];1.e7",
                aer_stream,
            ),
            IcBcMappingItem(
                "seas -> 1.0*[SS001]+1.0*[SS002]+1.0*[SS003]+1.0*[SS004]+1.0*[SS005];1.e9",
                aer_stream,
            ),
            IcBcMappingItem("ac0 -> 6.41*[BCPHOBIC]+6.41*[BCPHILIC];1.e11", aer_stream),
            IcBcMappingItem("ecj -> 1.0*[BCPHOBIC]+1.0*[BCPHILIC];1.e9", aer_stream),
            IcBcMappingItem("ac0 -> 6.41*[OCPHOBIC]+6.41*[OCPHILIC];1.e11", aer_stream),
            IcBcMappingItem("orgpaj -> 1.0*[OCPHOBIC]+1.0*[OCPHILIC];1.e9", aer_stream),
        ]

    if scheme_key == "gocart_dust":
        return [
            IcBcMappingItem(f"DUST_{k} -> 1.0*[DU00{k}];1.e9", aer_stream)
            for k in range(1, 6)
        ]

    if scheme_key == "gocart":
        return [
            *[IcBcMappingItem(f"DUST_{k} -> 1.0*[DU00{k}];1.e9", aer_stream) for k in range(1, 6)],
            *[IcBcMappingItem(f"SEAS_{k} -> 1.0*[SS00{k + 1}];1.e9", aer_stream) for k in range(1, 5)],
            IcBcMappingItem("so2 -> 0.453*[SO2];1.e6", aer_stream),
            IcBcMappingItem("sulf -> 0.302*[SO4];1.e6", aer_stream),
            IcBcMappingItem("BC1 -> 0.4143*[BCPHOBIC];1.e9", aer_stream),
            IcBcMappingItem("BC2 -> 0.4143*[BCPHILIC];1.e9", aer_stream),
            IcBcMappingItem("OC1 -> 0.4143*[OCPHOBIC];1.e9", aer_stream),
            IcBcMappingItem("OC2 -> 0.4143*[OCPHILIC];1.e9", aer_stream),
            IcBcMappingItem("dms -> 0.467*[DMS];1.e6", aer_stream),
        ]

    if scheme_key == "mosaic_8bin":
        return [
            IcBcMappingItem("oin_a02 -> 4.92e-4*[DU001];1.e9", aer_stream),
            IcBcMappingItem("oin_a03 -> 8.94e-3*[DU001];1.e9", aer_stream),
            IcBcMappingItem("oin_a04 -> 0.14300*[DU001];1.e9", aer_stream),
            IcBcMappingItem("oin_a05 -> 0.84740*[DU001]+0.1520*[DU002];1.e9", aer_stream),
            IcBcMappingItem("oin_a06 -> 0.84800*[DU002]+0.4055*[DU003];1.e9", aer_stream),
            IcBcMappingItem("oin_a07 -> 0.59450*[DU003]+0.4480*[DU004];1.e9", aer_stream),
            IcBcMappingItem("oin_a08 -> 0.55200*[DU004]+1.0*[DU005];1.e9", aer_stream),
            IcBcMappingItem("num_a02 -> 0.213e14*[DU001];1.0", aer_stream),
            IcBcMappingItem("num_a03 -> 0.598e14*[DU001];1.0", aer_stream),
            IcBcMappingItem("num_a04 -> 1.192e14*[DU001];1.0", aer_stream),
            IcBcMappingItem("num_a05 -> 1.434e14*[DU001]+0.948e13*[DU002];1.0", aer_stream),
            IcBcMappingItem("num_a06 -> 2.086e13*[DU002]+0.358e13*[DU003];1.0", aer_stream),
            IcBcMappingItem("num_a07 -> 0.255e13*[DU003]+0.592e12*[DU004];1.0", aer_stream),
            IcBcMappingItem("num_a08 -> 0.296e12*[DU004]+1.655e11*[DU005];1.0", aer_stream),
        ]

    raise ValueError(f"Unsupported MERRA-2 mapping scheme '{scheme}'")


def get_emac2wrf_mapping(scheme: str = "made_sorgam") -> list[IcBcMappingItem]:
    """
    EMAC (ECHAM5/MESSy `WRF_bc_chem` or legacy channel streams) -> WRF-Chem `chem_opt=100`.
    Includes fix for coarse sea-salt number term `2.01*[SS_cs]`.
    """
    scheme_key = str(scheme).lower()
    use_channels = scheme_key in ("messy_channels", "channels")
    if scheme_key not in ("made_sorgam", "default", "wrf_bc", "messy_channels", "channels"):
        raise ValueError(f"Unsupported EMAC mapping scheme '{scheme}'")

    def st(channel_name: str) -> str:
        return channel_name if use_channels else "WRF_bc_chem"

    poa_terms_ac0 = "+".join(
        [f"6.41*[fPOA0{k}_ks]" for k in range(1, 6)]
        + [f"6.41*[fPOA0{k}_as]" for k in range(1, 6)]
        + [f"6.41*[bbPOA0{k}_ks]" for k in range(1, 5)]
        + [f"6.41*[bbPOA0{k}_as]" for k in range(1, 5)]
    )
    poa_terms_mass = "+".join(
        [f"8.63*[fPOA0{k}_ks]" for k in range(1, 6)]
        + [f"8.63*[fPOA0{k}_as]" for k in range(1, 6)]
        + [f"8.63*[bbPOA0{k}_ks]" for k in range(1, 5)]
        + [f"8.63*[bbPOA0{k}_as]" for k in range(1, 5)]
    )

    return [
        IcBcMappingItem("so2 -> 1*[SO2];1.e6", st("tr_sulfur")),
        IcBcMappingItem("ch4 -> 1*[CH4];1.e6", st("tr_hycarbs")),
        IcBcMappingItem("co -> 1*[CO];1.e6", st("tr_alks")),
        IcBcMappingItem("o3 -> 1*[O3];1.e6", st("tr_Ox_HOx")),
        IcBcMappingItem("hono -> 1*[HONO];1.e6", st("tr_NOx_NOy")),
        IcBcMappingItem("hcho -> 1*[HCHO];1.e6", st("tr_alks")),
        IcBcMappingItem("eth -> 1*[C2H6];1.e6", st("tr_hycarbs")),
        IcBcMappingItem("hc3 -> 1*[C3H8];1.e6", st("tr_hycarbs")),
        # Sulfate (nucleation + accumulation)
        IcBcMappingItem("nu0 -> 2.99*[SO4mm_ns]+2.99*[SO4mm_ks];1.e14", st("aer_sulf")),
        IcBcMappingItem("so4ai -> 3.315*[SO4mm_ns]+3.315*[SO4mm_ks];1.e9", st("aer_sulf")),
        IcBcMappingItem("ac0 -> 3.56*[SO4mm_as]+3.56*[SO4mm_cs];1.e11", st("aer_sulf")),
        IcBcMappingItem("so4aj -> 3.315*[SO4mm_as]+3.315*[SO4mm_cs];1.e9", st("aer_sulf")),
        # Dust (coarse)
        IcBcMappingItem("corn -> 1.65*[DU_as]+1.65*[DU_ai]+1.65*[DU_cs]+1.65*[DU_ci];1.e7", st("aer_du_ss")),
        IcBcMappingItem("soila -> 1.38*[DU_as]+1.38*[DU_ai]+1.38*[DU_cs]+1.38*[DU_ci];1.e9", st("aer_du_ss")),
        # Sea salt (fixed missing '*' in 2.01*[SS_cs])
        IcBcMappingItem("corn -> 2.01*[SS_ks]+2.01*[SS_as]+2.01*[SS_cs];1.e7", st("aer_du_ss")),
        IcBcMappingItem("seas -> 2.017*[SS_ks]+2.017*[SS_as]+2.017*[SS_cs];1.e9", st("aer_du_ss")),
        # Black Carbon
        IcBcMappingItem("ac0 -> 3.2*[BC_ks]+3.2*[BC_as]+3.2*[BC_cs]+3.2*[BC_ki];1.e11", st("aer_bc_oc")),
        IcBcMappingItem("ecj -> 0.414*[BC_ks]+0.414*[BC_as]+0.414*[BC_cs]+0.414*[BC_ki];1.e9", st("aer_bc_oc")),
        # Primary Organic Aerosols
        IcBcMappingItem(f"ac0 -> {poa_terms_ac0};1.e11", st("aer_OA")),
        IcBcMappingItem(f"orgpaj -> {poa_terms_mass};1.e9", st("aer_OA")),
        # Nitrate
        IcBcMappingItem("nu0 -> 3.56*[NO3m_ns]+3.56*[NO3m_ks];1.e14", st("aer_nitr")),
        IcBcMappingItem("no3ai -> 2.14*[NO3m_ns]+2.14*[NO3m_ks];1.e9", st("aer_nitr")),
        IcBcMappingItem("ac0 -> 4.24*[NO3m_as]+4.24*[NO3m_cs];1.e11", st("aer_nitr")),
        IcBcMappingItem("no3aj -> 2.14*[NO3m_as]+2.14*[NO3m_cs];1.e9", st("aer_nitr")),
        # Ammonium
        IcBcMappingItem("nu0 -> 3.11*[NH4p_ns]+3.11*[NH4p_ks];1.e14", st("aer_nitr")),
        IcBcMappingItem("nh4ai -> 0.62*[NH4p_ns]+0.62*[NH4p_ks];1.e9", st("aer_nitr")),
        IcBcMappingItem("ac0 -> 3.70*[NH4p_as]+3.70*[NH4p_cs];1.e11", st("aer_nitr")),
        IcBcMappingItem("nh4aj -> 0.62*[NH4p_as]+0.62*[NH4p_cs];1.e9", st("aer_nitr")),
    ]


def get_mapping_for_source(source: str, scheme: str = "made_sorgam") -> list[IcBcMappingItem]:
    """Return mapping rule list for donor model `cams`, `merra2`, or `emac`."""
    src = str(source).lower()
    if src == "cams":
        return get_cams2wrf_mapping(scheme=scheme)
    if src == "merra2":
        return get_merra2wrf_mapping(scheme=scheme)
    if src == "emac":
        return get_emac2wrf_mapping(scheme=scheme)
    raise ValueError(f"Unsupported donor source '{source}'. Choose from ('cams', 'merra2', 'emac').")
