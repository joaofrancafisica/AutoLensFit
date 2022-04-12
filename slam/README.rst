The ``slam`` folder contains the Source, Light and Mass (SLaM) pipeline, which are **advanced** template pipelines for
automated lens modeling with **PyAutoLens**:

Files (Advanced)
----------------

- ``extensions.py``: Extensions to the SLaM pipelines that perform specific lens model fits.
- ``light_parametric.py``: The light pipeline which fits a model to the lens galaxy's light, which follows the source pipeline.
- ``mass_light_dark.py``: The mass pipeline which fits a ``light_dark`` mass model (e.g. separate mass profile for the stellar and dark matter) which follows the Light pipeline.
- ``mass_total.py``: The mass pipeline which fits a ``total`` mass model (e.g. a single mass profile for the stellar and dark matter) which follows the source or light pipeline.
- ``slum_util.py``: Utilities used in the SLaM pipelines.
- ``source_inversion.py``: The source pipeline which fits a pixelized source reconstruction which follows the source parametric pipeline.
- ``source_parametric.py``: The source pipeline which fits a parametric source model which starts a SLaM pipeline run.
- ``subhalo.py``: The subhalo pipeline which fits a dark matter substructure and follows a mass pipeline.