.PHONY: clear Debug Release

CMAKE := /opt/cmake-4.1.1/bin/cmake

format: ./src
	find src \
	-path src/ImGui -prune -o \
  -type f \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.glsl' \) \
  -exec clang-format -i {} +

clear:
	trash build && mkdir build && cd build

configure: clear
	${CMAKE} --preset aliu-configure

Debug:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cis5650_stream_compaction_test

Release:
	${CMAKE} --build --preset aliu-$@ && ./build/bin/$@/cis5650_stream_compaction_test