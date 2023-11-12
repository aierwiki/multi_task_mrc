import os
import re
import pdb


class SplitSentence(object):
    """ 分句，将文本切分为若干句子，其中处理引号的部分逻辑情况较多

    Args:
        text(str): 字符串文本
        criterion(coarse/fine): 句子切分粒度，粗细两种 `coarse` 或 `fine`，
            `coarse` 指的是按句号级别切分，`fine` 指按所有标点符合切分，
            默认按照粗粒度进行切分

    Returns:
        list(str): 分句后的句子列表

    Examples:
        >>> text = '中华古汉语，泱泱大国，历史传承的瑰宝。'
        >>> print(jio.split_sentence(text, criterion='fine'))

        # ['中华古汉语，', '泱泱大国，', '历史传承的瑰宝。']

    """
    def __init__(self):
        self.puncs_fine = None

    def _prepare(self):
        self.puncs_fine = {'……', '\r\n', '，', '。', ';', '；', '…', '！', '!',
                           '?', '？', '\r', '\n', '“', '”', '‘', '’', '：'}
        self.puncs_coarse = {'。', '！', '？', '\n', '“', '”', '‘', '’'}
        self.front_quote_list = {'“', '‘'}
        self.back_quote_list = {'”', '’'}

        self.puncs_coarse_ptn = re.compile('([。“”！？\n])')
        self.puncs_fine_ptn = re.compile('([，：。;“”；…！!?？\r\n])')

    def _split_long_sentence(self, sentence, max_sen_len=100):
        final_sentences = []
        tmp_sentences = self.puncs_fine_ptn.split(sentence)
        # 如果按照标点符号切分之后还是太长，就按照字数切分
        short_sentences = []
        for sen in tmp_sentences:
            if len(sen) > max_sen_len:
                for i in range(0, len(sen), max_sen_len):
                    short_sentences.append(sen[i:i+max_sen_len])
            else:
                short_sentences.append(sen)

        tmp_sentence = ''
        for sen in short_sentences:
            if len(tmp_sentence) + len(sen) > max_sen_len:
                final_sentences.append(tmp_sentence)
                tmp_sentence = sen
            else:
                tmp_sentence = tmp_sentence + sen
        if len(tmp_sentence) > 0:
            final_sentences.append(tmp_sentence)
        return final_sentences

    def _merge_short_sentences(self, sentences, min_sen_len=20):
        # 将句子长度小于min_sen_len的句子合并
        final_sentences = []
        tmp_sentence = ''
        for sen in sentences:
            if len(tmp_sentence) + len(sen) < min_sen_len:
                tmp_sentence = tmp_sentence + sen
            else:
                if len(tmp_sentence) > 0:
                    final_sentences.append(tmp_sentence)
                tmp_sentence = sen
        if len(tmp_sentence) > 0:
            final_sentences.append(tmp_sentence)
        return final_sentences

    def _split_and_merge(self, sentences, max_sen_len=100, min_sen_len=20):
        # 先把短句合并
        sentences = self._merge_short_sentences(sentences, min_sen_len)
        # 把长句拆成短句
        final_sentences = []
        for sen in sentences:
            if len(sen) > max_sen_len:
                tmp_sens = self._split_long_sentence(sen, max_sen_len)
                final_sentences.extend(tmp_sens)
            else:
                final_sentences.append(sen)
        return final_sentences


    def __call__(self, text, criterion='coarse', max_sen_len=None, min_sen_len=None):

        if self.puncs_fine is None:
            self._prepare()

        if criterion == 'coarse':
            tmp_list = self.puncs_coarse_ptn.split(text)
        elif criterion == 'fine':
            tmp_list = self.puncs_fine_ptn.split(text)
        else:
            raise ValueError('The parameter `criterion` must be '
                             '`coarse` or `fine`.')

        final_sentences = []
        quote_flag = False

        for sen in tmp_list:
            if sen == '':
                continue

            if criterion == 'coarse':
                if sen in self.puncs_coarse:
                    if len(final_sentences) == 0:  # 即文本起始字符是标点
                        if sen in self.front_quote_list:  # 起始字符是前引号
                            quote_flag = True
                        final_sentences.append(sen)
                        continue

                    # 以下确保当前标点前必然有文本且非空字符串
                    # 前引号较为特殊，其后的一句需要与前引号合并，而不与其前一句合并
                    if sen in self.front_quote_list:
                        if final_sentences[-1][-1] in self.puncs_coarse:
                            # 前引号前有标点如句号，引号等：另起一句，与此合并
                            final_sentences.append(sen)
                        else:
                            # 前引号之前无任何终止标点，与前一句合并
                            final_sentences[-1] = final_sentences[-1] + sen
                        quote_flag = True
                    else:  # 普通,非前引号，则与前一句合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    continue

            elif criterion == 'fine':
                if sen in self.puncs_fine:
                    if len(final_sentences) == 0:  # 即文本起始字符是标点
                        if sen in self.front_quote_list:  # 起始字符是前引号
                            quote_flag = True
                        final_sentences.append(sen)
                        continue

                    # 以下确保当前标点前必然有文本且非空字符串
                    # 前引号较为特殊，其后的一句需要与前引号合并，而不与其前一句合并
                    if sen in self.front_quote_list:
                        if final_sentences[-1][-1] in self.puncs_fine:
                            # 前引号前有标点如句号，引号等：另起一句，与此合并
                            final_sentences.append(sen)
                        else:
                            # 前引号之前无任何终止标点，与前一句合并
                            final_sentences[-1] = final_sentences[-1] + sen
                        quote_flag = True
                    else:  # 普通,非前引号，则与前一句合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    continue

            if len(final_sentences) == 0:  # 起始句且非标点
                final_sentences.append(sen)
                continue

            if quote_flag:  # 当前句子之前有前引号，须与前引号合并
                final_sentences[-1] = final_sentences[-1] + sen
                quote_flag = False
            else:
                if final_sentences[-1][-1] in self.back_quote_list:
                    # 此句之前是后引号，需要考察有无其他终止符，用来判断是否和前句合并
                    if len(final_sentences[-1]) <= 1:
                        # 前句仅一个字符。后引号，则合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    else:  # 前句有多个字符，
                        if criterion == 'fine':
                            if final_sentences[-1][-2] in self.puncs_fine:
                                # 有逗号等，则需要另起一句，该判断不合语文规范，但须考虑此情况
                                final_sentences.append(sen)
                            else:  # 前句无句号，则需要与前句合并
                                final_sentences[-1] = final_sentences[-1] + sen

                        elif criterion == 'coarse':
                            if final_sentences[-1][-2] in self.puncs_coarse:
                                # 有句号，则需要另起一句
                                final_sentences.append(sen)
                            else:  # 前句无句号，则需要与前句合并
                                final_sentences[-1] = final_sentences[-1] + sen
                else:
                    final_sentences.append(sen)

        # 增加切分长句合并短句的逻辑
        if max_sen_len is not None and min_sen_len is not None:
            final_sentences = self._split_and_merge(final_sentences, max_sen_len, min_sen_len)
        return final_sentences


if __name__ == '__main__':
    split_sentence = SplitSentence()
    text = '央视新闻消息，近日，特朗普老友皮尔斯·摩根喊话特朗普：“美国人的生命比你的选举更重要。如果你继续以自己为中心，继续玩弄愚蠢的政治……如果你意识不到自己>的错误，你就做不对”。目前，特朗普已“取关”了这位老友。'
    # res = split_sentence(text, criterion='fine')
    # res = split_sentence(text, criterion='coarse', max_sen_len=100, min_sen_len=20)
    res = split_sentence(text, criterion='fine', max_sen_len=100, min_sen_len=20)
    # res = split_sentence(text, criterion='coarse')
    print(res)