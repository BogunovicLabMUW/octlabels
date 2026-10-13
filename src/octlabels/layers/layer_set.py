"""One B-scan: metadata plus A-scan aligned layers and regions."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

from octlabels.groups import NamedGroup
from octlabels.info import INFO_KEYS
from octlabels.layers.annotation import AScanAnnotation, AScanRegion, Layer

A = TypeVar("A", bound=AScanAnnotation)

# Keywords of layerlib 0.5 that would otherwise be taken as custom properties.
_RENAMED_KEYWORDS = {
    "uncertainty_array": "uncertainty=uncertainty_from_std(std)",
    "layer_names": "names",
    "optional_info": "info",
}


class AnnotationGroup(NamedGroup[A]):
    """Ordered, name-keyed collection of one kind of A-scan annotation.

    Behaves like a dict. Every member must have the width of the owning
    [`LayerSet`][]; the first member sets ``info["width"]`` if it is missing.

    Args:
        kind: The annotation class stored in this group.
        owner: The LayerSet this group belongs to.
    """

    def __init__(self, kind: Type[A], owner: LayerSet) -> None:
        super().__init__(kind, check=self._check_width)
        self._owner = owner

    def _check_width(self, annotation: A) -> None:
        self._owner._claim_width(len(annotation))

    def property_dtypes(self) -> Dict[str, np.dtype]:
        """Returns every property name used by any member, with its dtype."""
        dtypes: Dict[str, np.dtype] = {}
        for annotation in self._items.values():
            for key, array in annotation.properties.items():
                dtypes.setdefault(key, array.dtype)
        return dtypes

    def stack(
        self, key: str = "values", names: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Stacks ``values`` or a property of the members into one array.

        Args:
            key: ``"values"`` or a property name.
            names: Members to stack, in this order. Defaults to all.

        Returns:
            Array of shape [num_members, width].

        Raises:
            KeyError: If a member or a member's property does not exist.
        """
        if names is None:
            names = list(self._items)
        if not names:
            if key == "values":
                dtype = self.kind.VALUES_DTYPE
            else:
                dtype = self.kind.PROPERTY_DTYPES.get(key, np.float32)
            return np.empty((0, self._owner.width or 0), dtype=dtype)

        rows = []
        for name in names:
            annotation = self[name]
            if key == "values":
                rows.append(annotation.values)
            elif key in annotation.properties:
                rows.append(annotation.properties[key])
            else:
                raise KeyError(
                    f"{self.kind.__name__} '{name}' has no property '{key}'."
                )
        return np.stack(rows)

    def unstack(self, array: ArrayLike, key: str = "values") -> None:
        """Sets ``values`` or a property of all members from a stacked array.

        Args:
            array: Array of shape [num_members, width], rows in member order.
            key: ``"values"`` or a property name.
        """
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[0] != len(self):
            raise ValueError(f"Expected shape ({len(self)}, width), got {array.shape}.")
        for annotation, row in zip(self._items.values(), array, strict=True):
            if key == "values":
                annotation.values = row
            else:
                annotation.set_property(key, row)


class LayerSet:
    """Annotations of one B-scan.

    Attributes:
        info: Metadata such as ``width``, ``height`` and ``idx``.
        layers: Layer boundaries, see [`AnnotationGroup`][].
        regions: Biomarker regions, see [`AnnotationGroup`][].
    """

    SECTIONS: ClassVar[Tuple[str, ...]] = ("layers", "regions")
    """Annotation sections, in file order. A new annotation kind needs an attribute in
    ``__init__`` and an entry here.
    """

    def __init__(self, info: Optional[Mapping[str, Any]] = None) -> None:
        self.info: Dict[str, Any] = dict(info or {})
        self.layers: AnnotationGroup[Layer] = AnnotationGroup(Layer, self)
        self.regions: AnnotationGroup[AScanRegion] = AnnotationGroup(AScanRegion, self)

    def sections(self) -> Dict[str, AnnotationGroup]:
        """Returns all annotation groups by section name."""
        return {section: getattr(self, section) for section in self.SECTIONS}

    @property
    def width(self) -> Optional[int]:
        """Number of A-scans, from ``info["width"]``."""
        return self.info.get("width")

    @property
    def height(self) -> Optional[int]:
        """Number of pixels per A-scan, from ``info["height"]``."""
        return self.info.get("height")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """``(num_layers, height, width)``, missing dimensions are 0."""
        return (len(self.layers), self.height or 0, self.width or 0)

    def _claim_width(self, width: int) -> None:
        if self.width is None:
            self.info["width"] = width
        elif self.width != width:
            raise ValueError(
                f"Width {width} does not match LayerSet width {self.width}."
            )

    def __len__(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def __contains__(self, name: str) -> bool:
        """Whether a layer with this name exists."""
        return name in self.layers

    def __getitem__(self, key: Union[int, str, slice, Tuple[Any, slice]]) -> Any:
        """Returns a layer by position or name, or a LayerSet cropped along
        the A-scan axis for ``ls[a:b]`` or ``ls[:, a:b]``.
        """
        if isinstance(key, numbers.Integral):
            return list(self.layers.values())[key]
        if isinstance(key, str):
            return self.layers[key]
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], slice):
            key = key[1]
        if isinstance(key, slice):
            return self._crop(key)
        raise TypeError(f"Invalid key type for LayerSet: {type(key)}.")

    def _crop(self, columns: slice) -> LayerSet:
        """Returns a copy with only the A-scans selected by ``columns``."""

        def crop(annotation: AScanAnnotation) -> AScanAnnotation:
            return annotation.map_arrays(lambda key, array: array[columns])

        new_width = len(range(*columns.indices(self.width or 0)))
        return self.map_annotations(crop, info={**self.info, "width": new_width})

    def __repr__(self) -> str:
        return (
            f"LayerSet(info={self.info}, layers={list(self.layers)}, "
            f"regions={list(self.regions)})"
        )

    def __str__(self) -> str:
        parts = ["--- Info ---"]
        parts += [f"  {key}: {value}" for key, value in self.info.items()]
        for section, group in self.sections().items():
            if group:
                parts.append(f"\n--- {section.capitalize()} ({len(group)}) ---")
                parts.extend(str(annotation) for annotation in group.values())
        return "\n".join(parts)

    def map_annotations(
        self,
        func: Callable[[AScanAnnotation], AScanAnnotation],
        info: Optional[Mapping[str, Any]] = None,
    ) -> LayerSet:
        """Returns a new LayerSet with ``func`` applied to every annotation.

        Args:
            func: Maps an annotation to a new one; may change the width.
            info: Metadata of the result. Defaults to a copy of ``self.info``.
                ``width`` is taken from the new annotations if there are any.
        """
        mapped = {}
        widths = set()
        for section, group in self.sections().items():
            mapped[section] = [func(annotation) for annotation in group.values()]
            widths.update(len(annotation) for annotation in mapped[section])
        if len(widths) > 1:
            raise ValueError(f"Mapped annotations have different widths: {widths}.")

        result = LayerSet(self.info if info is None else info)
        if widths:
            result.info["width"] = widths.pop()
        for section, members in mapped.items():
            for annotation in members:
                getattr(result, section).add(annotation)
        return result

    def copy(self) -> LayerSet:
        """Returns a deep copy."""
        return self.map_annotations(lambda annotation: annotation.copy())

    def _select(
        self,
        names: Optional[Sequence[str]],
        region_names: Optional[Sequence[str]],
        create_missing: bool,
    ) -> LayerSet:
        result = LayerSet(self.info)
        wanted_by_section = {"layers": names, "regions": region_names}
        for section, source in self.sections().items():
            target = getattr(result, section)
            wanted = wanted_by_section[section]
            if wanted is None:
                wanted = list(source)
            dtypes = source.property_dtypes()
            for name in wanted:
                if name in source:
                    target.add(source[name].copy())
                elif create_missing:
                    target.add(source.kind.zeros(name, self.width or 0, dtypes))
                else:
                    raise KeyError(f"{source.kind.__name__} '{name}' not found.")
        return result

    def subset(
        self, names: Sequence[str], region_names: Optional[Sequence[str]] = None
    ) -> LayerSet:
        """Returns a copy with only the given layers (and regions, if given;
        all regions otherwise). Raises KeyError for unknown names."""
        return self._select(names, region_names, create_missing=False)

    def ensure(
        self, names: Sequence[str], region_names: Optional[Sequence[str]] = None
    ) -> LayerSet:
        """Returns a copy with exactly the given layers, in this order.

        Missing layers are created with zero values and zero properties (the
        union of properties of the existing layers), so stacking works. The
        same applies to ``region_names``; if it is None, all regions are kept.
        """
        return self._select(names, region_names, create_missing=True)

    def add_layer(self, layer: Layer) -> None:
        """Adds a layer. Raises ValueError for duplicate names or wrong width."""
        self.layers.add(layer)

    def add_layer_raw(
        self,
        name: str,
        values: ArrayLike,
        confidence: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
        **properties: ArrayLike,
    ) -> None:
        """Creates and adds a layer from arrays."""
        self.layers.add(Layer(name, values, confidence, uncertainty, **properties))

    def add_region(self, region: AScanRegion) -> None:
        """Adds a region. Raises ValueError for duplicate names or wrong width."""
        self.regions.add(region)

    def add_region_raw(
        self,
        name: str,
        values: ArrayLike,
        confidence: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
        **properties: ArrayLike,
    ) -> None:
        """Creates and adds a region from arrays."""
        self.regions.add(
            AScanRegion(name, values, confidence, uncertainty, **properties)
        )

    def values_to_array(self) -> np.ndarray:
        """Layer values with NaN replaced by 0. Shape: [num_layers, width]."""
        return np.nan_to_num(self.layers.stack("values"))

    def confidence_to_array(self) -> np.ndarray:
        """Layer confidences. Shape: [num_layers, width]."""
        return self.layers.stack("confidence")

    def uncertainty_to_array(self) -> np.ndarray:
        """Layer uncertainties. Raises KeyError if a layer has none."""
        return self.layers.stack("uncertainty")

    def to_array(
        self, names: Optional[Sequence[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(values, confidence)`` of the layers, each
        [num_layers, width]; NaN values become 0.

        Args:
            names: Layers to return, in this order; missing ones are zero
                (value 0, confidence 0). Defaults to all layers.
        """
        if names is None:
            layer_set = self
        else:
            layer_set = self.ensure(names, region_names=[])
        return layer_set.values_to_array(), layer_set.confidence_to_array()

    def values_from_array(self, values: ArrayLike) -> None:
        """Sets layer values from an array of shape [num_layers, width]."""
        self.layers.unstack(values, "values")

    def confidence_from_array(self, confidence: ArrayLike) -> None:
        """Sets layer confidences from an array of shape [num_layers, width]."""
        self.layers.unstack(confidence, "confidence")

    def uncertainty_from_array(self, uncertainty: ArrayLike) -> None:
        """Sets layer uncertainties from an array of shape [num_layers, width]."""
        self.layers.unstack(uncertainty, "uncertainty")

    def update_from_array(
        self,
        values: ArrayLike,
        confidence: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
    ) -> None:
        """Sets layer values and, if given, confidences and uncertainties."""
        self.values_from_array(values)
        if confidence is not None:
            self.confidence_from_array(confidence)
        if uncertainty is not None:
            self.uncertainty_from_array(uncertainty)

    @classmethod
    def from_array(
        cls,
        values: ArrayLike,
        confidence: Optional[ArrayLike] = None,
        names: Optional[Sequence[str]] = None,
        info: Optional[Mapping[str, Any]] = None,
        **properties: ArrayLike,
    ) -> LayerSet:
        """Creates a LayerSet from stacked layer arrays.

        Args:
            values: Boundary heights. Shape: [num_layers, width].
            confidence: Confidences, same shape. Defaults to all ones.
            names: Layer names. Defaults to ``"Layer 0"``, ``"Layer 1"``, ...
            info: Metadata, e.g. ``{"height": 496}``.
            **properties: Further properties, each [num_layers, width], e.g.
                ``uncertainty=uncertainty_from_std(std)``.
        """
        for old_name, new_name in _RENAMED_KEYWORDS.items():
            if old_name in properties:
                raise TypeError(f"'{old_name}' was renamed; use {new_name}.")
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {values.shape}.")
        num_layers = values.shape[0]
        if names is None:
            names = [f"Layer {i}" for i in range(num_layers)]
        if len(names) != num_layers:
            raise ValueError(f"Got {len(names)} names for {num_layers} layers.")
        if confidence is not None:
            confidence = np.asarray(confidence)
        properties = {key: np.asarray(array) for key, array in properties.items()}

        # width first, as in files written by hand and by older versions
        layer_set = cls({"width": values.shape[1], **(info or {})})
        for i, name in enumerate(names):
            layer_properties = {key: array[i] for key, array in properties.items()}
            layer_confidence = None if confidence is None else confidence[i]
            layer_set.add_layer_raw(
                name, values[i], layer_confidence, **layer_properties
            )
        return layer_set

    @classmethod
    def empty(
        cls,
        names: Sequence[str],
        width: int,
        info: Optional[Mapping[str, Any]] = None,
        region_names: Optional[Sequence[str]] = None,
    ) -> LayerSet:
        """Creates a LayerSet with zero-filled layers and regions."""
        layer_set = cls({"width": width, **(info or {})})
        return layer_set.ensure(names, region_names or [])

    def validate(self) -> None:
        """Checks metadata and value ranges.

        Raises:
            ValueError: If width or height is missing, a key from
                [`INFO_KEYS`][] has an invalid value, an
                annotation does not match the width, or a layer lies outside
                [0, height].
        """
        for key in ("width", "height"):
            if key not in self.info:
                raise ValueError(f"info['{key}'] is missing.")
        for key, value in self.info.items():
            known = INFO_KEYS.get(key)
            if known is not None and not known.is_valid(value):
                raise ValueError(
                    f"info['{key}'] = {value!r} is invalid. {known.description}"
                )
        for group in self.sections().values():
            for annotation in group.values():
                if len(annotation) != self.width:
                    raise ValueError(
                        f"'{annotation.name}' has width {len(annotation)}, "
                        f"expected {self.width}."
                    )
        for layer in self.layers.values():
            if np.any(layer.values < 0) or np.any(layer.values > self.height):
                raise ValueError(
                    f"Layer '{layer.name}' has values outside [0, height]."
                )

    # The file format lives in octlabels.layers.serialization, which imports
    # this module; the methods below import it on use to avoid a cycle.

    def to_string(self, yaml_arrays: bool = False) -> str:
        """Serializes to the layer file format.

        See [`serialization`][octlabels.layers.serialization].
        """
        from octlabels.layers.serialization import dumps

        return dumps(self, yaml_arrays=yaml_arrays)

    @classmethod
    def from_string(cls, text: str) -> LayerSet:
        """Parses the layer file format.

        See [`serialization`][octlabels.layers.serialization].
        """
        from octlabels.layers.serialization import loads

        return loads(text)

    def save(
        self,
        path: Union[str, Path],
        compress: Optional[str] = None,
        yaml_arrays: bool = False,
    ) -> Path:
        """Writes the LayerSet to a file.

        See [`save`][octlabels.layers.serialization.save].
        """
        from octlabels.layers.serialization import save

        return save(self, path, compress=compress, yaml_arrays=yaml_arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> LayerSet:
        """Reads a ``.yml``, ``.yml.gz`` or ``.yml.xz`` file."""
        from octlabels.layers.serialization import load

        return load(path)
