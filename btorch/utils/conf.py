from pathlib import Path

from omegaconf import OmegaConf


def load_config(Param, use_config_file=True, search_path=Path(".")):
    """Doesn't support help text and Literal though."""
    defaults = OmegaConf.structured(Param)
    cli_cfg = OmegaConf.from_cli()
    if use_config_file and "config_path" in cli_cfg:
        assert "config_path" not in Param.__dataclass_fields__
        config_path = Path(cli_cfg.config_path)
        if not config_path.is_file():
            config_path = search_path / config_path
            assert config_path.is_file()
        cfg_cli_file = OmegaConf.load(cli_cfg.config_path)
        cli_cfg.pop("config_path")
    else:
        cfg_cli_file = OmegaConf.create()
    cfg = OmegaConf.unsafe_merge(defaults, cfg_cli_file, cli_cfg)
    cfg = OmegaConf.to_object(cfg)
    return cfg
