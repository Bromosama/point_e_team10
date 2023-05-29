% ----- sampler.py ---- %
Sampler -> PointCloudSampler:
	CALL sampler.sample_batch_progressive(batch: 1, model_kwargs (img: [256, 256]))
		output: (x -> samples: [1, 6, 1024])
        % Makes a call to embedd the image
        CALL model.cached_model_kwargs(batch_size, stage_model_kwargs)
            output: embeddings: [1, 1024, 256]
        CALL karras_sample_progressive(diffusion, model, samle_shape)
            % ----- k_diffusion.py ---- %
            % Define the random noise point cloud
            x_T: [1, 6, 1024] 
            Define model = GaussianToKarrasDenoiser: 
            CALL gudied_denoised(x_t):
                % concat x_t to x_t
                x_t: [2, 6, 1024]
                model.denoise(x_T, sigma, image_embeddings)
                    t: [2]
                    c_in: [2, 1, 1]
                    % Define X for variance and mean of gaussian 
                    % X: [N: number of parameters, C: pointcloud parameters, T: number of points in pointcloud]
                    x: x_t*c_in: [2, 6, 1024]
                    % ----- gaussian_diffusion.py ---- %
                    CALL p_mean_variance(x, t, model)
                        output: out['pred_xstart'] 
                        B, C = [2], [6]
                        % ----- transformer.py ---- %
                        CALL model(x, t, image_embeddings) % CLIPImageGridPointDiffusionTransformer
                            x: [2, 6, 1024]
                            t: [2]
                            time_token_cond: True 
                            img_embeddings: [2, 1024, 256]
                            img_embeddings_permuded: [2, 256, 1024]
                            t_embeddings: [2, 512]
                            % we use 12 blocks?
                            output: [2, 12, 1024]
                            cond = [t_emb: [2, 512], True, clip_emb[2, 256, 1024], True]
                            CALL forward_with_cond(x, cond)
                                x: [2, 6, 1024]
                                % x is permuted and thrown into a linear projection layer
                                h <- x: [2, 1024, 512]
                                % Defines from cond_token
                                extra_tokens: [[[2, 1, 512], [2, 256, 512]]]
                                % Concatenates to h
                                h <- h, ekstra_tokens: [2, 1281, 512]
                                h <- ln_pre(h): [2, 1281, 512]
                                % Calling resblock
                                h <- backbone(h): [2, 1281, 512]
                                h <- ln_post(h): [2, 1281, 512]
                                h <- sum(h): [2, 1024, 512]
                                h <- output_proj(h): [2, 1024, 12]
                                h <- permute(h): [2, 12, 1024]
                                output h
                        % ----- gaussian_diffusion.py ---- %
                        % Splits output h: model_output, model_var_values <- model_output (on dim 1)
                        model_output: [2, 6, 1024]
                        model_var_values: [2, 6, 1024]
                        process_xstart(x.clamp <- model_output): 
                        outputs: dict{mean: [2, 6, 1024], variance: [2, 6, 1024], pred_xstart: [2, 6, 1024]}
                    % ----- k_diffusion.py ---- %
                    outputs: denoised <- pred_xstart
                x_0 <- denoised [2, 6, 1024]
                cond x [1, 6, 1024], uncond x [1, 6, 1024] <- x_0 ()
                outputs x0: [1, 6, 1024]       
