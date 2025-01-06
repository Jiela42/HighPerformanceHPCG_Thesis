#include <thrust/system/cpp/memory.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/detail/generic/tag.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/vector_base.h>
#include <thrust/detail/type_traits.h>

namespace amgx {
    namespace thrust {
        using host_system_tag = ::thrust::system::cpp::tag;
        using device_system_tag = ::thrust::system::cuda::tag;
        using any_system_tag = ::thrust::system::detail::generic::tag;

        template <typename T>
        using device_ptr = ::thrust::device_ptr<T>;

        template <typename T>
        using device_reference = typename ::thrust::device_reference<T>;

        template <typename T, typename Alloc>
        using device_vector = ::thrust::device_vector<T, Alloc>;

        template <typename Iterator>
        using iterator_reference = typename ::thrust::iterator_reference<Iterator>::type;

        template <typename Iterator>
        using iterator_difference = typename ::thrust::iterator_difference<Iterator>::type;

        template <typename Iterator>
        using iterator_value = typename ::thrust::iterator_value<Iterator>::type;

        template <typename Iterator>
        using iterator_system = typename ::thrust::iterator_system<Iterator>::type;

        template <typename Iterator>
        using iterator_pointer = typename ::thrust::iterator_pointer<Iterator>::type;

        namespace detail {
            using ::thrust::detail::vector_base;
            using ::thrust::detail::eval_if;
            using ::thrust::detail::vector_equal;
        }
    }
}