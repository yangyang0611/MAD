from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class CustomDDIMScheduler(DDIMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.1,
        use_clipped_model_output: bool = False,
        generator=None,
        original_image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        std_dev_t = eta * self._get_variance(timestep, prev_timestep) ** (0.5)
        noise = (
            randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            if variance_noise is None
            else variance_noise
        )

        variance = 0
        if timestep > 0 and eta > 0:
            variance = std_dev_t * noise

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if original_image is not None and mask is not None:
            prev_unknown_part = (
                (alpha_prod_t_prev**0.5) * pred_original_sample + pred_sample_direction + variance
            )

            # 7-1. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
            prev_known_part = (alpha_prod_t_prev**0.5) * original_image + (
                ((1.0 - alpha_prod_t_prev) ** 0.5) * (noise if timestep > 0 else 0)
            )

            # 7-2. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
            prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part
        else:
            prev_sample = (
                (alpha_prod_t_prev**0.5) * pred_original_sample + pred_sample_direction + variance
            )

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def compute_eps(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        next_sample: torch.FloatTensor,
        eta: float = 0.1,
    ) -> torch.FloatTensor:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # 8. Compute the added noise
        variance_noise = (next_sample - prev_sample) / (std_dev_t if eta > 0 else 1)
        return variance_noise

    def set_inverse_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)[::-1]
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def ddim_inverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        # 1. get previous step value (=t+1)
        prev_timestep = timestep
        timestep = min(
            timestep - self.config.num_train_timesteps // self.num_inference_steps,
            self.config.num_train_timesteps - 1,
        )

        # 2. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )


class CustomDDPMScheduler(DDPMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        original_image: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        *args,
        **kwargs,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf

        std_dev_t = self._get_variance(timestep, predicted_variance) ** (0.5)
        noise = (
            randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            if variance_noise is None
            else variance_noise
        )

        variance = 0
        if timestep > 0:
            variance = std_dev_t * noise

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if original_image is not None and mask is not None:
            prev_unknown_part = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * sample
                + variance
            )

            # 7-1. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
            prev_known_part = (alpha_prod_t_prev**0.5) * original_image + (
                ((1.0 - alpha_prod_t_prev) ** 0.5) * (noise if timestep > 0 else 0)
            )

            # 7-2. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
            pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part
        else:
            pred_prev_sample = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * sample
                + variance
            )

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )

    def compute_eps(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        next_sample: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        assert timestep > 0, "computing eps for ddpm is only available for t > 0"
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )

        std_eps = self._get_variance(t, predicted_variance=predicted_variance) ** 0.5

        variance_noise = (next_sample - pred_prev_sample) / (std_eps if std_eps > 0 else 1)
        return variance_noise

    def set_inverse_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            )
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)[::-1]
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def ddim_inverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        # 1. get previous step value (=t+1)
        prev_timestep = timestep
        timestep = min(
            timestep - self.config.num_train_timesteps // self.num_inference_steps,
            self.config.num_train_timesteps - 1,
        )

        # 2. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.alphas_cumprod[0]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
