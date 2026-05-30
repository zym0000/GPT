import torch
import copy

class YiConversationRender:

    def __init__(self,tokenizer,device):
        self.device = device
        #Yi特殊token
        self.bos_token_id = tokenizer.bos_token_id or 1
        self.eos_token_id = tokenizer.eos_token_id or 2
        self.user_start = '<|user_start|>'
        self.user_end = '<|user_end|>'
        self.assistant_start = '<|assistant_start|>'
        self.assistant_end = '<|assistant_end|>'
        self.FIM_PREFIX = '<fim_prefix>'
        self.FIM_SUFFIX = '<fim_suffix>'
        self.FIM_MIDDLE = '<fim_middle>'

        self.add_token =[
            self.user_start,
            self.user_end,
            self.assistant_start,
            self.assistant_end,
            self.FIM_PREFIX,
            self.FIM_SUFFIX,
            self.FIM_MIDDLE,
        ]

        tokenizer.add_special_tokens({
                "additional_special_tokens":  self.add_token
        })

        current = len(tokenizer)  # 63992 或 63996
        target = 64000
        dummy_tokens = [f"||<|reserved_{i}|>" for i in range(target - current)]
        tokenizer.add_special_tokens({"additional_special_tokens": dummy_tokens})

        self.tokenizer = tokenizer

        self.imstart_id = tokenizer.convert_tokens_to_ids(self.user_start)
        self.imend_id = tokenizer.convert_tokens_to_ids(self.user_end)
        self.assistant_start_id = tokenizer.convert_tokens_to_ids(self.assistant_start)
        self.assistant_end_id = tokenizer.convert_tokens_to_ids(self.assistant_end)
        self.FIM_PREFIX_ID = tokenizer.convert_tokens_to_ids(self.FIM_PREFIX)
        self.FIM_SUFFIX_ID = tokenizer.convert_tokens_to_ids(self.FIM_SUFFIX)
        self.FIM_MIDDLE_ID = tokenizer.convert_tokens_to_ids(self.FIM_MIDDLE)
        self._init_token_bytes()

    def get_bos_id(self):
        return  self.bos_token_id
    
    def get_eos_id(self):
        return self.eos_token_id
    
    def get_model_max_length(self):
        return self.tokenizer.model_max_length
    
    def get_fim_ids(self):
        return self.FIM_PREFIX_ID,self.FIM_SUFFIX_ID,self.FIM_MIDDLE_ID
    
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids,skip_special_tokens = False):
        return self.tokenizer.decode(token_ids,skip_special_tokens)
    
    def get_vocab_size(self):
        return len(self.tokenizer)
    
    def get_token_bytes(self):
        return self.token_bytes
    
    def get_doc_batch_tokens(self,doc_batch):
        #这里使用这种方式，做并行处理，要比下面的批量处理方式速度快
        #backend_tokenizer  只负责把文本编程 token,使用下面的方式，在python阶段会进行model_max_len 检查
        encodings = self.tokenizer.backend_tokenizer.encode_batch(doc_batch)
    
        # enc.ids 返回纯 token ID 列表（不含特殊 token）
        token_list = [enc.ids for enc in encodings]
        return token_list
       # return self.tokenizer(doc_batch,add_special_tokens=False)["input_ids"]
    
    def _init_token_bytes(self):
        """
            获取tokenizer token_bytes
        """
        vocab_size = self.get_vocab_size()
        self.token_bytes  = torch.zeros(vocab_size,dtype=torch.int64,device=self.device)
        special_ids = set()

         # Yi 特有的
        for attr in ['pad_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id']:
            val = getattr(self.tokenizer,attr,None)

            if val is not None:
                special_ids.add(val)

        for token_str in self.add_token:
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            special_ids.add(token_id)

        for token_id in range(vocab_size):
            if token_id in special_ids:
                continue

            text = self.tokenizer.decode([token_id],skip_special_tokens=False)
            self.token_bytes[token_id] = len(text.encode("utf-8"))


    def render_conversation_complete(self, conversation):
        conversation = copy.deepcopy(conversation)
        message = conversation["messages"]
        assert message[-1]["role"] == "assistant"
        message.pop()

        ids,mask = self.render_conversation(conversation)
        ids.append(self.assistant_start_id)
        return torch.tensor(ids, dtype=torch.long)
        
    """
    {"messages": [{"role": "user", "content": "Write a function in Python to split a given string into the words in the string."}, 
    {"role": "assistant", "content": "def split_string(string):\n    return string.split()"}], 
    "source": "alpaca"}
    格式如下
    <BOS>
    <|user_start|>system
    You are a helpful assistant.<|user_end|>
    <|user_start|>user
    Hello<|user_end|>
    <|assistant_start|>assistant
    Hi there<|assistant_end|>
    <EOS>
    """
    def render_conversation(self, conversation, max_token = 4096):
        ids = []
        mask = []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
        
        if conversation['messages'][0]['role'] == 'user':
            messages = conversation['messages']

        assert len(messages) >= 1
        bos = self.get_bos_id()

        add_tokens(bos, 0)

        for i,message in enumerate(messages):
            role = message['role']
            content = message['content']

            if role == 'user':
                add_tokens(self.imstart_id,0)
                content_ids = self.encode(content)
                add_tokens(content_ids,0)
                add_tokens(self.imend_id,0)
            elif role == 'assistant':
                add_tokens(self.assistant_start_id,0)
                if isinstance(content,str):
                    content_ids = self.encode(content)
                    add_tokens(content_ids,1)
                elif isinstance(content,list):
                    """
                    {"type":"python","text": expr}
                    {"type":"python_output","text":result}
                    {"type":"text","text":part}
                    """
                    for part in content:
                        type = part.get('type','text')
                        text = part.get('type','')
                        value_id = self.encode(text)

                        if type == 'text':
                            add_tokens(value_id,1)
                        elif type == 'python_output':
                            value_id = self.encode(text)
                            add_tokens(value_id,0)
                        elif type == 'python':
                            add_tokens(value_id,1)
                        else:
                            raise ValueError(f"Unknown part type: {type}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                
                add_tokens(self.assistant_end_id,0)
            else:
                 raise ValueError(f"Unknown role: {role}")
        
        while len(ids) > 0 and ids[-1] == self.eos_token_id:
            ids.pop()
            mask.pop()
        
        add_tokens(self.eos_token_id,1)

        # 3. 最终截断
        ids = ids[:max_token]
        mask = mask[:max_token]

        if len(ids) >= max_token and ids[-1] != self.eos_token_id:
            # 去掉最后一个 token，腾出空间给 EOS
            ids = ids[:max_token-1]
            mask = mask[:max_token-1]
            ids.append(self.eos_token_id)
            mask.append(1)

        max_id = max(ids) if ids else 0

        if max_id > 64000:
            print(f"!!! render_conversation 越界 !!!")
            print(f"max_id={max_id}, vocab_size=63992")
            print(f"ids={ids[:30]}...")
            # 找到是哪个部分引入的
            for i, tid in enumerate(ids):
                if tid > 64000:
                    print(f"  位置 {i}: id={tid}, 前面上下文={ids[max(0,i-5):i]}")
                    print(self.tokenizer.decode([63998]))
                
        return ids, mask
