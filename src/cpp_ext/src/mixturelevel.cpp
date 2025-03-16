#include "mixturelevel.hpp"
#include "featurevector.hpp"
#include "vec.hpp"

namespace py = pybind11;

namespace hem
{
	MixtureLevel::MixtureLevel(): pointSet(), covarianceSet(), colorSet(), opacities(), features()
	{

	}

	MixtureLevel MixtureLevel::CreateMixtureLevel(
	const py::list& xyz,
    const py::list& colors,
    const py::list& opacities,
    const py::list& covariance,
    const py::list& features)
    {
        hem::MixtureLevel mixtureLevel;
        mixtureLevel.pointSet = xyz.cast<std::vector<vec3>>();
        mixtureLevel.colorSet = colors.cast<std::vector<vec3>>();
        mixtureLevel.opacities = opacities.cast<std::vector<float>>();
        mixtureLevel.covarianceSet = covariance.cast<std::vector<smat3>>();
        mixtureLevel.features = features.cast<std::vector<FeatureVector>>();
        return mixtureLevel;
    }

    pybind11::tuple MixtureLevel::CreatePythonLists(MixtureLevel &mixtureLevel)
    {
        py::list xyz;
        for (const auto& point : mixtureLevel.pointSet) {
            py::list point_list;
            point_list.append(point.x);
            point_list.append(point.y);
            point_list.append(point.z);
            xyz.append(point_list);
        }

        py::list colors;
        for (const auto& color : mixtureLevel.colorSet) {
            py::list color_list;
            color_list.append(color.x);
            color_list.append(color.y);
            color_list.append(color.z);
            colors.append(color_list);
        }

        py::list opacities = py::cast(mixtureLevel.opacities);

        py::list covariance;
        for (const auto& cov : mixtureLevel.covarianceSet) {
            py::list cov_list;
            cov_list.append(cov.e00);
            cov_list.append(cov.e01);
            cov_list.append(cov.e02);
            cov_list.append(cov.e11);
            cov_list.append(cov.e12);
            cov_list.append(cov.e22);
            covariance.append(cov_list);
        }

        py::list features;
        for (const auto& feature : mixtureLevel.features) {
            features.append(py::cast(feature.GetVector()));
        }

        return py::make_tuple(xyz, colors, opacities, covariance, features);
    }
}