#include <gtest/gtest.h>

#include <nnpack.h>
#include <nnpack/macros.h>
#include <nnpack/hwinfo.h>

#include <testers/convolution.h>

/*
 * Test that implementation works for a single tile of transformation
 */

TEST(FT16x16, single_tile) {
	ConvolutionTester()
		.inputSize(8, 8)
		.iterations(10)
		.errorLimit(1.0e-5)
		.testInference(nnp_convolution_algorithm_ft16x16, nnp_activation_identity);
}

int main(int argc, char* argv[]) {
	const enum nnp_status init_status = nnp_initialize();
	assert(init_status == nnp_status_success);
	setenv("TERM", "xterm-256color", 0);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}