import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import ContextGem components
from contextgem.public.documents import Document
from contextgem.public.llms import DocumentLLM
from contextgem.public.aspects import Aspect
from contextgem.public.concepts import StringConcept
from contextgem.public.images import Image

# Import PDF processing and LiteLLM
import litellm
from PIL import Image as PILImage
import fitz  # PyMuPDF for PDF to image conversion
import base64
import io

# Configure LiteLLM to use Qwen API
litellm.set_verbose = True

def convert_pdf_to_contextgem_images(pdf_path: str):
    """Convert PDF pages to ContextGem Image objects"""
    try:
        contextgem_images = []
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]

            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality

            # Convert to PIL Image then to bytes
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # Convert to base64
            base64_data = base64.b64encode(img_bytes).decode('utf-8')

            # Create ContextGem Image with correct parameters
            contextgem_img = Image(
                mime_type="image/png",
                base64_data=base64_data
            )
            contextgem_images.append(contextgem_img)

        pdf_document.close()
        return contextgem_images
    except Exception as e:
        print(f"PDF转ContextGem图像错误: {str(e)}")
        return None


def define_extraction_aspects():
    """Define the aspects and concepts for PDF extraction - focused on testing basis and results"""

    # Create aspect for testing basis and results extraction
    aspect = Aspect(
        name="检测依据与结果提取",
        description="专门提取文档中所有与检测依据（标准、规范、方法）和检测结果（数据、结论、参数）相关的内容",
        concepts=[
            StringConcept(
                name="检测依据",
                description="文档中提到的所有检测标准、规范、方法、规程等依据性文件和条款"
            ),
            StringConcept(
                name="检测标准",
                description="具体的国家标准、行业标准、地方标准或企业标准等检测标准信息"
            ),
            StringConcept(
                name="检测方法",
                description="使用的具体检测方法、测量方法、试验方法等"
            ),
            StringConcept(
                name="检测结果",
                description="所有的检测结果数据、测量数据、测试数据等定量或定性结果"
            ),
            StringConcept(
                name="检测数据",
                description="具体的数值检测结果，包括测量值、误差、不确定度等"
            ),
            StringConcept(
                name="检测结论",
                description="基于检测结果得出的结论、判定、评估意见等"
            ),
            StringConcept(
                name="检测参数",
                description="检测过程中涉及的各种技术参数、条件参数等"
            ),
            StringConcept(
                name="设备信息",
                description="检测用设备的型号、规格、编号等信息"
            )
        ],
        llm_role="extractor_text",
        reference_depth="paragraphs",
        add_justifications=True,
        justification_depth="brief"
    )

    return aspect

async def extract_content_with_two_step(pdf_path: str):
    """Two-step extraction: Qwen VL OCR + ContextGem structured extraction"""
    try:
        # Step 1: Use Qwen VL for OCR to extract text
        print("=== 步骤1: 使用Qwen VL进行文字提取 ===")
        ocr_text = await extract_pdf_text_with_vision_ocr_simple(pdf_path)

        if not ocr_text:
            raise ValueError("文字提取失败，无法获得文本内容")

        print(f"文字提取完成，共 {len(ocr_text)} 字符")

        # Step 2: Use ContextGem for structured extraction
        print("\n=== 步骤2: 使用ContextGem进行结构化提取 ===")

        # Create document from OCR text
        print("正在创建ContextGem文档对象...")
        document = Document(raw_text=ocr_text)
        print(f"文档对象创建完成，包含 {len(document.paragraphs)} 个段落")

        # Define extraction aspects
        print("正在定义提取规则...")
        aspect = define_extraction_aspects()

        # Create ContextGem LLM for text extraction
        print("正在配置ContextGem文本模型...")
        llm = create_qwen_vl_llm()

        # Assign aspect to document
        document = document.clone()
        document.add_aspects([aspect])

        # Use ContextGem's structured extraction
        print("开始执行ContextGem结构化提取...")
        try:
            result = await llm.extract_aspects_from_document_async(document)
            print("ContextGem提取操作完成")

            # Display results
            print("\n=== 两步结构化提取结果 ===")
            print(f"处理的文件: {pdf_path}")
            print(f"提取的方面: {aspect.name}")
            print("-" * 60)

            # Get the processed aspect from result
            processed_aspect = result[0] if result else aspect

            # Display extracted items
            extracted_items = processed_aspect.extracted_items
            print(f"提取到的项目数量: {len(extracted_items)}")

            if extracted_items:
                print(f"成功提取到 {len(extracted_items)} 项结构化信息：")

                # Group by concept for better organization
                concept_groups = {}
                for item in extracted_items:
                    if hasattr(item, 'concept') and hasattr(item.concept, 'name'):
                        concept_name = item.concept.name
                    else:
                        concept_name = '未知概念'

                    if concept_name not in concept_groups:
                        concept_groups[concept_name] = []
                    concept_groups[concept_name].append(item)

                # Display each concept group
                for concept_name, items in concept_groups.items():
                    print(f"\n📋 {concept_name}:")
                    print("-" * 40)
                    for i, item in enumerate(items, 1):
                        value = getattr(item, 'value', str(item))
                        justification = getattr(item, 'justification', None)

                        print(f"{i}. {value}")
                        if justification:
                            print(f"   理由: {justification}")
                        print()
            else:
                print("ContextGem未提取到结构化信息")

            print("-" * 60)
            return result

        except Exception as extract_error:
            print(f"ContextGem提取失败: {str(extract_error)}")
            return None

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_qwen_vl_llm():
    """Create a Qwen VL (Vision) LLM configuration using LiteLLM"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not found in environment variables")

    # Create LLM configuration for ContextGem using Qwen VL
    llm = DocumentLLM(
        model="dashscope/qwen-vl-plus",  # Qwen VL model
        api_key=api_key,
        api_base=base_url,
        temperature=0.1,
        max_tokens=8000,
        role="extractor_text"  # Use text role - ContextGem will handle images automatically
    )

    # Manually set vision capability since LiteLLM doesn't detect it
    llm._supports_vision = True

    return llm

async def extract_pdf_text_with_vision_ocr_simple(pdf_path: str):
    """Simple OCR using Qwen VL to extract text from PDF images"""
    try:
        print(f"开始处理PDF文件: {pdf_path}")

        # Convert PDF to ContextGem images
        print("正在转换PDF为图像...")
        images = convert_pdf_to_contextgem_images(pdf_path)

        if not images:
            raise ValueError("无法将PDF转换为图像")

        print(f"成功转换PDF为 {len(images)} 张图像")

        # Create Qwen VL LLM for simple OCR
        print("正在配置Qwen VL模型进行文字提取...")
        llm = create_qwen_vl_llm()

        # Process all images at once for OCR (simpler approach)
        ocr_prompt = """
请从这些检测报告图像中提取所有文字内容。

要求：
1. 提取所有文字，保持原始格式
2. 识别表格中的数据
3. 保持数字、符号、单位等
4. 按照页面顺序整理内容

请完整提取文字内容，不要遗漏任何信息。
"""

        try:
            # Use Qwen VL for direct text extraction
            result = await llm.chat_async(ocr_prompt, images=images)

            print(f"\n=== PDF OCR文字提取完成 ===")
            print(f"处理页数: {len(images)}")
            print(f"提取文本长度: {len(result)} 字符")
            print("-" * 40)
            print("提取的前200个字符:")
            print(result[:200] + "..." if len(result) > 200 else result)
            print("-" * 40)

            return result

        except Exception as extract_error:
            print(f"OCR提取失败: {str(extract_error)}")
            return None

    except Exception as e:
        print(f"OCR处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run PDF extraction using Qwen VL + ContextGem two-step approach"""

    # PDF file path
    pdf_path = "A224005962110101E.pdf"

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"错误: PDF文件 '{pdf_path}' 不存在")
        return

    # Run extraction
    print("=== Qwen VL OCR + ContextGem 两步结构化提取工具 ===")

    try:
        # Run the async function
        import asyncio
        result = asyncio.run(extract_content_with_two_step(pdf_path))

        if result:
            print("✅ 两步结构化提取完成！")
        else:
            print("❌ 两步结构化提取失败！")

    except Exception as e:
        print(f"程序执行错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
