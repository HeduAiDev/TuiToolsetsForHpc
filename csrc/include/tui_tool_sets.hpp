#pragma once

#include <stddef.h>    // for size_t
#include <array>       // for array
#include <atomic>      // for atomic
#include <chrono>      // for operator""s, chrono_literals
#include <cmath>       // for sin
#include <functional>  // for ref, reference_wrapper, function
#include <memory>      // for allocator, shared_ptr, __shared_ptr_access
#include <string>  // for string, basic_string, char_traits, operator+, to_string
#include <thread>   // for sleep_for, thread
#include <utility>  // for move
#include <vector>   // for vector
#include "ftxui/component/component.hpp"  // for Checkbox, Renderer, Horizontal, Vertical, Input, Menu, Radiobox, ResizableSplitLeft, Tab
#include "ftxui/component/component_base.hpp"  // for ComponentBase, Component
#include "ftxui/component/component_options.hpp"  // for MenuOption, InputOption
#include "ftxui/component/event.hpp"              // for Event, Event::Custom
#include "ftxui/component/screen_interactive.hpp"  // for Component, ScreenInteractive
#include "ftxui/dom/elements.hpp"  // for text, color, operator|, bgcolor, filler, Element, vbox, size, hbox, separator, flex, window, graph, EQUAL, paragraph, WIDTH, hcenter, Elements, bold, vscroll_indicator, HEIGHT, flexbox, hflow, border, frame, flex_grow, gauge, paragraphAlignCenter, paragraphAlignJustify, paragraphAlignLeft, paragraphAlignRight, dim, spinner, LESS_THAN, center, yframe, GREATER_THAN
#include "ftxui/dom/flexbox_config.hpp"  // for FlexboxConfig
#include "ftxui/screen/color.hpp"  // for Color, Color::BlueLight, Color::RedLight, Color::Black, Color::Blue, Color::Cyan, Color::CyanLight, Color::GrayDark, Color::GrayLight, Color::Green, Color::GreenLight, Color::Magenta, Color::MagentaLight, Color::Red, Color::White, Color::Yellow, Color::YellowLight, Color::Default, Color::Palette256, ftxui
#include "ftxui/screen/color_info.hpp"  // for ColorInfo
#include "ftxui/screen/terminal.hpp"    // for Size, Dimensions
#ifdef __CUDA__
#include <cuda_fp16.h>
#endif

namespace tui {
    namespace component {
        using namespace ftxui;

        struct Resizable4BlockOptions {
            Element placeholder_block1 = nullptr;
            Element placeholder_block2 = nullptr;
            Element placeholder_block3 = nullptr;
            Element placeholder_block4 = nullptr;
            std::function<Element()> separator_func = [] { return separator(); };
            std::function<Element()> separator_hover_func = [] { return separatorHeavy(); };
            Ref<int> base_x = -1;
            Ref<int> base_y = -1;
            Resizable4BlockOptions() = default;
            // Custom copy constructor
            Resizable4BlockOptions(const Resizable4BlockOptions& other)
                : placeholder_block1(other.placeholder_block1),
                placeholder_block2(other.placeholder_block2),
                placeholder_block3(other.placeholder_block3),
                placeholder_block4(other.placeholder_block4),
                separator_func(other.separator_func),
                separator_hover_func(other.separator_hover_func),
                base_x(std::move(other.base_x)),
                base_y(std::move(other.base_y)) {}
            };

        class Resizable4Blockbase : public ftxui::ComponentBase {
            public:
                enum class Direction {
                    Horizontal,
                    Vertical
                };
                // block1 | block2
                // block3 | block4
                explicit Resizable4Blockbase(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options = Resizable4BlockOptions());
                Element Render() override;
                bool OnEvent(Event event) override;
                bool OnMouseEvent(Event event);
                bool isDragging();
                Element getVSeparator();
                Element getHSeparator();
            private:
                Component block1_;
                Component block2_;
                Component block3_;
                Component block4_;
                ScreenInteractive& screen_;
                Box vseparator_up_box_;
                Box vseparator_down_box_;
                Box hseparator_box_;
                bool is_hover_hseparator_ = false;
                bool is_hover_vseparator_up_ = false;
                bool is_hover_vseparator_down_ = false;
                bool is_dragging_hseparator_ = false;
                bool is_dragging_vseparator_up_ = false;
                bool is_dragging_vseparator_down_ = false;
                int bias_x_ = 0;
                int bias_y_ = 0;
                Resizable4BlockOptions options_;
        };

        struct RadioFrameOptions : public RadioboxOption {
            std::string title_regx = "selected: %s";
            int max_width = 200;
            int max_height = 200;
            int min_width = 0;
            int min_height = 0;
            RadioFrameOptions() = default;
        };

        class RadioFrameBase : public ftxui::ComponentBase, public RadioFrameOptions {
            public:
                explicit RadioFrameBase(RadioFrameOptions& options);
                Element Render() override;
            private:
                Component content_;
        };




        enum class InputType {
            Text,
            Password,
            Number
        };
        
        struct InputElementConfig : public InputOption {
            ftxui::StringRef label;
            InputType input_type = InputType::Text;
            int max_label_width = -1;
            int min_label_width = -1;
            int max_input_width = -1;
            int min_input_width = -1;
            std::function<Element(Element)> label_style = nullptr;
            std::function<Element(Element)> input_style = nullptr;
        };
        struct InputFormOptions {
            using ElementRowConfig = std::vector<InputElementConfig>;
            InputType default_input_type = InputType::Text;
            int default_max_label_width = 200;
            int default_min_label_width = 0;
            int default_max_input_width = 200;
            int default_min_input_width = 0;
            std::vector<ElementRowConfig> elements_config;
            std::function<Element(Element)> default_label_style = [] (Element ele) { return ele; };
            std::function<Element(Element)> default_input_style = [] (Element ele) { return ele; };
            InputFormOptions() = default;
        };
        class InputFormBase : public ftxui::ComponentBase, public InputFormOptions {
            public:
                explicit InputFormBase(InputFormOptions& options);
                Element Render() override;
            private:
                Component setWidth(Component component, int max_width, int min_width);
                Element setWidth(Element element, int max_width, int min_width);
                std::vector<Component> renderFormRow(ElementRowConfig row);
                std::vector<std::vector<Component>> components_;
        };

        struct MatrixFrameOptionsCommonElementStyle
        {
            using ElementStyle = ::std::function<void(::ftxui::Element &ele, int x, int y, ::ftxui::Element &separator_right, ::ftxui::Element &separator_bottom, ::ftxui::Element &separator_cross)>;
            // using Decorator = ::std::function<ElementStyle(ElementStyle)>;

            static ::std::function<ElementStyle()> empty_style;
            static ::std::function<ElementStyle(int row_id, Color color)> hight_light_row;
            static ::std::function<ElementStyle(int col_id, Color color)> hight_light_col;
            static ::std::function<ElementStyle(int row_id, int col_id, Color trace_color, Color point_color)> mark_point_trace;
            static ::std::function<ElementStyle(int row_id, int col_id, Color color)> mark_point;
            static ::std::function<ElementStyle(int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color)> mark_sub_matrix;

            // friend MatrixFrameOptionsCommonElementStyle::ElementStyle operator|(MatrixFrameOptionsCommonElementStyle::ElementStyle lhs, MatrixFrameOptionsCommonElementStyle::ElementStyle rhs);
        };

        struct MatrixFrameOptionsLabelMark {
            enum class LabelType {
                Row,
                Col
            };
            LabelType type;
            int id;
            Color color = Color::Gold3Bis;
            Color bgcolor = Color::Grey3;
            MatrixFrameOptionsLabelMark(LabelType type, int id, Color color = Color::Gold3Bis, Color bgcolor = Color::Grey3)
            : type(type), id(id), color(color), bgcolor(bgcolor) {};
            MatrixFrameOptionsLabelMark(std::string type, int id, Color color = Color::Gold3Bis, Color bgcolor = Color::Grey3)
            : type(type == "row" ? LabelType::Row : LabelType::Col), id(id), color(color), bgcolor(bgcolor) {};
        };

        template <typename T>
        struct MatrixFrameOptions {
            T* ptr;
            int rows;
            int cols;
            // │ele│
            // ┼───┼ , separator_right: |, separator_bottom: ───, separator_cross: ┼
            // usage: options.element_style =  MatrixFrameOptionsCommonElementStyle::empty_style() | MatrixFrameOptionsCommonElementStyle::mark_point_trace(10, 20, Color::Blue1, Color::Red1);
            MatrixFrameOptionsCommonElementStyle::ElementStyle element_style = nullptr;
            // due to the element_style is a function, so we need to store the element_style in a stack limited to the max depth of the recursion
            ::std::vector<MatrixFrameOptionsCommonElementStyle::ElementStyle> element_style_stack;
            ::std::vector<MatrixFrameOptionsLabelMark> label_marks;
            Ref<float> focus_x = new float(0.5f);
            Ref<float> focus_y = new float(0.5f); 
            MatrixFrameOptions() = default;
            MatrixFrameOptions(const MatrixFrameOptions& other)
            : ptr(other.ptr), rows(other.rows), cols(other.cols), element_style(other.element_style), focus_x(other.focus_x), focus_y(other.focus_y), label_marks(std::move(other.label_marks)), element_style_stack(std::move(other.element_style_stack)) {};
        };

        template <typename T>
        class MatrixFrameBase: public ftxui::ComponentBase, public MatrixFrameOptions<T> {
            public:
                explicit MatrixFrameBase(MatrixFrameOptions<T>& options);
                Element Render() override;
                Element getColLabels();
                Element getRowLabels();
                Element getMatrix();
                float& getFocusX();
                float& getFocusY();
                friend tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle operator|( tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle lhs, tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle rhs);
            private:
                Element col_labels_;
                Element row_labels_;
                Component slider_x_;
                Component slider_y_;
                Element matrix_;
                int text_width_ = 3;
                
        };


        Component Resizable4Block(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options = Resizable4BlockOptions());
        Component RadioFrame(RadioFrameOptions options = RadioFrameOptions());
        Component RadioFrame(ConstStringListRef entries, int* selected, RadioFrameOptions options = RadioFrameOptions());
        Component InputForm(std::vector<InputFormOptions::ElementRowConfig> elements_config, InputFormOptions options = InputFormOptions());


        Component MatrixFrame(float* ptr, int rows, int cols, MatrixFrameOptions<float> options = MatrixFrameOptions<float>());
        Component MatrixFrame(double* ptr, int rows, int cols, MatrixFrameOptions<double> options = MatrixFrameOptions<double>());
        Component MatrixFrame(int* ptr, int rows, int cols, MatrixFrameOptions<int> options = MatrixFrameOptions<int>());
        #ifdef __CUDA__
        Component MatrixFrame(half* ptr, int rows, int cols, MatrixFrameOptions<half> options = MatrixFrameOptions<half>());
        #endif

    }
}
tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle operator|( tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle lhs, tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle rhs);
// {
//     return [lhs, rhs](ftxui::Element &ele, int x, int y, ftxui::Element &separator_right, ftxui::Element &separator_bottom, ftxui::Element &separator_cross) {
//         lhs(ele, x, y, separator_right, separator_bottom, separator_cross);
//         rhs(ele, x, y, separator_right, separator_bottom, separator_cross);
//     };
// }