void initFP8Scales(float max_input)
    {
        check_cuda_error(cudaStreamSynchronize(mStream->get()));

        // Add shift to the max because we add an adjustment for each expert so they get different results.
        float max_shift = expertShift(mNumExperts - 1, mNumExperts);
        float maxW1 = max_shift + (mIsGated ? std::max(mExpertWDiag1, mExpertWDiagGated) : mExpertWDiag1);
        float maxW2 = max_shift + mExpertWDiag2;
        float scaleW1 = getFP8Scalar(maxW1);
        float scaleW2 = getFP8Scalar(maxW2);
        mFP8WeightScalar1 = scaleW1;
        mFP8WeightScalar2 = scaleW2;

        float scaleAct1 = getFP8Scalar(max_input);

        float maxFC1Output = calcMLPVal(max_input, mNumExperts - 1) / maxW2;
        float scaleAct2 = getFP8Scalar(maxFC1Output);

        ASSERT_NE(mExpertFP8Scale1, nullptr);
        ASSERT_NE(mExpertFP8Scale2, nullptr);
        ASSERT_NE(mExpertFP8Scale3, nullptr);

        // Dequant values for each expert are 1/(w_i*a_i) calculated above
        std::vector<float> scales_1(mNumExperts, 1.f / (scaleW1 * scaleAct1));
        std::vector<float> scales_2(1, scaleAct2);
        std::vector<float> scales_3(mNumExperts, 1.f / (scaleW2 * scaleAct2));

        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale1, scales_1.data(), scales_1.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale2, scales_2.data(), scales_2.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));
        check_cuda_error(cudaMemcpyAsync(mExpertFP8Scale3, scales_3.data(), scales_3.size() * sizeof(float),
            cudaMemcpyHostToDevice, mStream->get()));

        check_cuda_error(cudaStreamSynchronize(mStream->get()));
    }