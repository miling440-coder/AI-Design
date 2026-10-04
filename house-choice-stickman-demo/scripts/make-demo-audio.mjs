import { mkdirSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

// 快速学习 Demo：把一条完整口播切成与场景对应的 6 段，
// 再根据原文短句长度生成近似字幕时码。正式项目可把这里替换成 ASR 词级时码。
const source = ".media/audio/voice/voice_001.mp3";
const frames = [
  {
    frame: 1, start: 0, duration: 19,
    phrases: ["你有没有发现", "很多年轻人不是买不起房", "而是——不敢不买房", "二十七八岁", "工作四五年", "自己攒了二三十万", "父母再拿出一点积蓄", "差不多就够一套房子的首付了", "然后你会发现", "周围几乎所有人都开始告诉你"],
  },
  {
    frame: 2, start: 19, duration: 20,
    phrases: ["早点买", "房子迟早都要有", "没房以后怎么结婚", "等以后涨了你就更买不起了", "最可怕的是什么", "是听久了以后", "你自己也开始慌", "这套房子适不适合现在的我", "如果我不买", "我是不是就落后了"],
  },
  {
    frame: 3, start: 39, duration: 20,
    phrases: ["很多年前", "也有一个年轻人面临类似的选择", "他工作没几年", "手里刚刚攒下人生第一笔钱", "这笔钱可以拿去买一套房", "但是如果买了", "自己手里的资本基本也就没有了", "还有另外一个选择", "把钱留下来", "继续做自己真正擅长的事情", "妻子选择了后者", "这个年轻人叫沃伦·巴菲特"],
  },
  {
    frame: 4, start: 59, duration: 19,
    phrases: ["但我讲这个故事", "不是想告诉你年轻人千万不要买房", "完全不是", "有些人需要稳定的住所", "有人有孩子有老人", "有人就是喜欢拥有一个真正属于自己的家", "房子当然有它的价值", "真正值得想的是另外一个问题", "当你人生第一次拥有二三十万的时候", "这笔钱对现在的你来说", "最重要的价值是什么"],
  },
  {
    frame: 5, start: 78, duration: 22,
    phrases: ["是一套房", "是一门技能", "一次创业机会", "换一座城市", "还是给自己几年时间", "去成为一个更有选择权的人", "很多时候", "我们花光积蓄购买的", "可能不仅仅是一套房子", "还有一种感觉", "我终于和别人一样了", "可人生真正危险的", "从来不是和别人不一样"],
  },
  {
    frame: 6, start: 100, duration: 18.944,
    phrases: ["而是有一天你突然发现", "你拼命做出的那些选择", "其实从来没有真正问过自己", "所以买房没有错", "不买也没有错", "真正值得害怕的是", "你根本不知道自己为什么要买", "如果现在给你30万", "一边是房子的首付", "一边是未来五年的可能性", "你会怎么选"],
  },
];

mkdirSync(".media/audio/voice/scenes", { recursive: true });

function timingFor(frame) {
  const gap = 0.22;
  const head = 0.18;
  const usable = frame.duration - head - gap * (frame.phrases.length - 1) - 0.16;
  const weights = frame.phrases.map((text) => Math.max(2, [...text].length));
  const total = weights.reduce((a, b) => a + b, 0);
  let cursor = head;
  return frame.phrases.map((text, i) => {
    const span = usable * weights[i] / total;
    const word = { id: `f${frame.frame}p${i + 1}`, text, start: +cursor.toFixed(3), end: +(cursor + span).toFixed(3) };
    cursor += span + gap;
    return word;
  });
}

const voices = [];
for (const frame of frames) {
  const path = `.media/audio/voice/scenes/${String(frame.frame).padStart(2, "0")}.wav`;
  const result = spawnSync("ffmpeg", [
    "-y", "-v", "error", "-ss", String(frame.start), "-i", source,
    "-t", String(frame.duration), "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", path,
  ], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
  voices.push({ frame: frame.frame, path, duration_s: frame.duration, words: timingFor(frame) });
}

writeFileSync("audio_meta.json", JSON.stringify({ bgm: null, bgm_pending: false, voices, sfx: [] }, null, 2) + "\n");
console.log(`created ${voices.length} voice clips and audio_meta.json`);
