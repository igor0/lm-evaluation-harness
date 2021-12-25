import transformers
import torch
from lm_eval.base import BaseLM
import numpy as np

from transformer_utils.logit_lens.hooks import make_lens_hooks
from transformer_utils.logit_lens.layer_names import make_layer_names


class HFLM(BaseLM):

    def __init__(self, device='cuda', pretrained='gpt2', revision='main', logit_lens=None, subfolder=None, tokenizer=None, batch_size=1):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        assert logit_lens in [None, "on", "off"]

        logit_lens_enabled = logit_lens == "on"

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained, revision=revision + ("/" + subfolder if subfolder is not None else "")
        ).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        assert isinstance(self.tokenizer, (
            transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
            transformers.T5Tokenizer, transformers.T5TokenizerFast,
        )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
            assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373], \
                self.tokenizer.encode('hello\n\nhello')

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

        if logit_lens_enabled:
            self.layer_names = make_layer_names(self.gpt2)
            make_lens_hooks(self.gpt2, layer_names=self.layer_names)
        else:
            self.layer_names = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        self.gpt2._last_resid = None
        with torch.no_grad():
            out = self.gpt2(inps)[0][:, :, :50257]
        self.gpt2._last_resid = None

        return self.extend_response(out)

    def _model_generate(self, context, max_length, eos_token_id):
        self.gpt2._last_resid = None
        out = self.gpt2.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False)
        self.gpt2._last_resid = None

        return self.extend_response(out)

    def extend_response(self, out):
        if self.layer_names is None:
            # no logit lens
            return out, None

        # collect the logit lens values.
        # TODO: requires quite a bit of contiguous memory; can be optimized
        layer_logits = np.stack(
            [self.gpt2._layer_logits[name] for name in self.layer_names],
        ) # [layer, batch, seq, vocab]

        layer_logits = np.swapaxes(layer_logits, 0, 1) # [batch, layer, seq, vocab]
        return out, layer_logits

# for backwards compatibility
GPT2LM = HFLM
