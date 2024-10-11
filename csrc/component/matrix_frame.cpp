#include <unordered_map>  // for unordered_map
#include "tui_tool_sets.hpp"
namespace tui {
    namespace component {
        using namespace ftxui;
        
        template<typename T>
        MatrixFrameBase<T>::MatrixFrameBase(MatrixFrameOptions<T>& options) : MatrixFrameOptions<T>(options) {
            // col_labels_ = getColLabels();
            // row_labels_ = getRowLabels();
            SliderOption<float> slider_x_option = {this -> focus_x, 0.0f, 1.0f, std::max(1/(float)this -> rows, 0.000001f), Direction::Right, Color::White, Color::Grey50};
            SliderOption<float> slider_y_option = {this -> focus_y, 0.0f, 1.0f, std::max(1/(float)this -> cols, 0.000001f), Direction::Down, Color::White, Color::Grey50};
            slider_x_ = Slider(slider_x_option) | bgcolor(Color::Grey23);
            slider_y_ = Slider(slider_y_option) | bgcolor(Color::Grey23);
            // matrix_ = getMatrix();
            Add(Container::Vertical({
                slider_x_,
                slider_y_,
            }));
        }


        template<typename T>
        Element MatrixFrameBase<T>::Render() {
            this -> updateBuffer();
            return hbox({
                hbox({
                    vbox({
                        slider_x_ -> Render() | size(HEIGHT, EQUAL, 1),
                        gridbox({
                            {getColLabels() | focusPositionRelative(this -> computeRelativeFocusX(), 0) | xframe },
                            {getMatrix() | focusPositionRelative(this -> computeRelativeFocusX(), this -> computeRelativeFocusY()) | frame },
                        }) ,
                    }) | xflex, 
                    vbox({
                        text(" ") | size(HEIGHT, EQUAL, 2),
                        hbox({
                            getRowLabels() | focusPositionRelative(0, this -> computeRelativeFocusY()) | yframe,
                            slider_y_ -> Render()
                        }) | yflex,
                    }) | size(WIDTH, GREATER_THAN, 4)
                }),
                hbox({text(" ")}) | xflex
            });
        }
        
        template<typename T>
        Element MatrixFrameBase<T>::getColLabels() {
            ::std::vector<Element> col_labels_arr;
            ::std::unordered_map<int, ::std::pair<Color,Color>> color_map;
            if (this -> label_marks.size() != 0) {
                for (MatrixFrameOptionsLabelMark& item : this -> label_marks) {
                    if (item.type == MatrixFrameOptionsLabelMark::LabelType::Col) {
                        color_map[item.id] = {item.color, item.bgcolor};
                    }
                }
            }
            for (int i = this -> buffer_x_(); i < std::min(this -> cols, this ->buffer_x_() + this -> buffer_cols_); i ++) {
                Color font_color, bg_color;
                ::std::tie(font_color, bg_color) = color_map.count(i) ? color_map[i] : ::std::pair<Color, Color>(Color::Gold3Bis, Color::Grey3);
                col_labels_arr.push_back(text(::std::to_string(i)) | center | frame | size(WIDTH, EQUAL, this -> text_width) | color(font_color) | bgcolor(bg_color));
                col_labels_arr.push_back(separator() | color(Color::Gold3) | bgcolor(Color::Grey3));
            }
            return gridbox({col_labels_arr});
        }

        template<typename T>
        Element MatrixFrameBase<T>::getRowLabels() {
            ::std::vector<::std::vector<Element>> row_labels_arr;
            ::std::unordered_map<int, ::std::pair<Color,Color>> color_map;
            if (this -> label_marks.size() != 0) {
                for (auto& [type, id, color, bgcolor] : this -> label_marks) {
                    if (type == MatrixFrameOptionsLabelMark::LabelType::Row) {
                        color_map[id] = {color, bgcolor};
                    }
                }
            }
            Color font_color, bg_color;
            for (int i = this -> buffer_y_(); i < std::min(this ->rows - 1, this ->buffer_y_() + this -> buffer_rows_ - 1); i ++) {
                ::std::tie(font_color, bg_color) = color_map.count(i) ? color_map[i] : ::std::pair<Color, Color>(Color::Gold3Bis, Color::Grey3);
                row_labels_arr.push_back({text(::std::to_string(i)) | size(HEIGHT, EQUAL, 1) | center | color(font_color) | bgcolor(bg_color)});
                row_labels_arr.push_back({separator() | color(Color::Gold3) | bgcolor(Color::Grey3)});
            }
            int i = std::min(this -> rows - 1, this ->buffer_y_() + this -> buffer_rows_ - 1);
            ::std::tie(font_color, bg_color) = color_map.count(i) ? color_map[i] : ::std::pair<Color, Color>(Color::Gold3Bis, Color::Grey3);
            row_labels_arr.push_back({text(::std::to_string(i)) | size(HEIGHT, EQUAL, 1) | center | color(font_color) | bgcolor(bg_color)});
            return gridbox(row_labels_arr);
        }

        #ifdef __CUDA__
        template <>
        Element  MatrixFrameBase<half>::getMatrix() {
            this -> updateBuffer();
            ::std::vector<Elements> _rows_arr;
            for (int i = this -> buffer_y_(); i < this -> buffer_y_() + this -> buffer_rows_; i++) {
                ::std::vector<Element> _cols_arr;
                ::std::vector<Element> _separator_arr;
                for (int j = this -> buffer_x_(); j < this -> buffer_x_() + this -> buffer_cols_; j++) {
                    float val = __half2float(this -> ptr[i * this -> cols + j]);
                    // │ele│
                    // ┼───┼
                    Element ele = text(std::to_string(val)) | center | frame | size(WIDTH, EQUAL, this -> text_width) | size(HEIGHT, EQUAL, 1); 
                    // |
                    Element separator_right = separator();
                    // ───
                    Element separator_bottom = separator();
                    // ┼
                    Element separator_cross = separator();
                    if (this -> element_style != nullptr) {
                        this -> element_style(ele, j, i, separator_right, separator_bottom, separator_cross);
                    } else if (this -> element_style_stack.size() != 0) {
                        for (auto &ele_style : this -> element_style_stack) {
                            ele_style(ele, j, i, separator_right, separator_bottom, separator_cross);
                        }
                    }
                    _cols_arr.push_back(ele);
                    _cols_arr.push_back(separator_right);
                    _separator_arr.push_back(separator_bottom);
                    _separator_arr.push_back(separator_cross);
                }
                _rows_arr.push_back(_cols_arr);
                if (i != this -> buffer_y_() + this -> buffer_rows_ - 1) {
                    _rows_arr.push_back(_separator_arr);
                }
            }
            return gridbox(_rows_arr);
        }
        #endif        


        template <typename T>
        Element  MatrixFrameBase<T>::getMatrix() {
            this -> updateBuffer();
            ::std::vector<Elements> _rows_arr;
            for (int i = this -> buffer_y_(); i < std::min(this -> rows, this -> buffer_y_() + this -> buffer_rows_); i++) {
                ::std::vector<Element> _cols_arr;
                ::std::vector<Element> _separator_arr;
                for (int j = this -> buffer_x_(); j < std::min(this -> cols, this -> buffer_x_() + this -> buffer_cols_); j++) {
                    T val = this -> ptr[i * this -> cols + j];
                    // │ele│
                    // ┼───┼
                    Element ele = text(std::to_string(val)) | center | frame | size(WIDTH, EQUAL, this -> text_width) | size(HEIGHT, EQUAL, 1); 
                    // |
                    Element separator_right = separator();
                    // ───
                    Element separator_bottom = separator();
                    // ┼
                    Element separator_cross = separator();
                    if (this -> element_style != nullptr) {
                        this -> element_style(ele, j, i, separator_right, separator_bottom, separator_cross);
                    } else if (this -> element_style_stack.size() != 0) {
                        for (auto &ele_style : this -> element_style_stack) {
                            ele_style(ele, j, i, separator_right, separator_bottom, separator_cross);
                        }
                    }
                    _cols_arr.push_back(ele);
                    _cols_arr.push_back(separator_right);
                    _separator_arr.push_back(separator_bottom);
                    _separator_arr.push_back(separator_cross);
                }
                _rows_arr.push_back(_cols_arr);
                if (i != std::min(this -> rows -1, this -> buffer_y_() + this -> buffer_rows_ - 1)) {
                    _rows_arr.push_back(_separator_arr);
                }
            }
            return gridbox(_rows_arr);
        }

        template<typename T>
        void MatrixFrameBase<T>::updateBuffer() {
            // buffer_x_ is X axis starting point of the buffer
            this -> buffer_x_ = this -> focus_x() * std::max(this -> cols - this -> buffer_cols_, 0);
            this -> buffer_y_ = this -> focus_y() * std::max(this -> rows - this -> buffer_rows_, 0);
        }

        template<typename T>
        float MatrixFrameBase<T>::computeRelativeFocusX() {
             
            float relative_bias = std::max((this -> focus_x() * this -> cols) - this -> buffer_x_(), 0.f);
            return relative_bias / (float)std::min(this -> cols, this -> buffer_cols_);
        }

        template<typename T>
        float MatrixFrameBase<T>::computeRelativeFocusY() {
             
            float relative_bias = std::max((this -> focus_y() * this -> rows) - this -> buffer_y_(), 0.f);
            return relative_bias / (float)std::min(this -> rows, this -> buffer_rows_);
        }

        template<typename T>
        float& MatrixFrameBase<T>::getFocusX() {
            return this -> focus_x;
        }

        template<typename T>
        float& MatrixFrameBase<T>::getFocusY() {
            return this -> focus_y;
        }

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, Color color)> MatrixFrameOptionsCommonElementStyle::hight_light_row = [](int row_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (y == row_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_right |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int col_id, Color color)> MatrixFrameOptionsCommonElementStyle::hight_light_col = [](int col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x == col_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, int col_id, Color color)> MatrixFrameOptionsCommonElementStyle::mark_point = [](int row_id, int col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x == col_id && y == row_id) {
                    ele |= ::ftxui::bgcolor(color);
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id - 1 && y == row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id && y == row_id - 1) {
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                if (x == col_id - 1 && y == row_id - 1) {
                    separator_cross |= ::ftxui::bgcolor(color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int row_id, int col_id, Color trace_color, Color point_color)> MatrixFrameOptionsCommonElementStyle::mark_point_trace = [](int row_id, int col_id, Color trace_color, Color point_color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [row_id, col_id, trace_color, point_color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {

                mark_point(row_id, col_id, point_color)(ele, x, y, separator_right, separator_bottom, separator_cross);
                if ( x > col_id && y == row_id ) {
                    ele |= ::ftxui::bgcolor(trace_color);
                    separator_right |= ::ftxui::bgcolor(trace_color);
                }
                if ( x == col_id && y < row_id ) {
                    ele |= ::ftxui::bgcolor(trace_color);
                    separator_bottom |= ::ftxui::bgcolor(trace_color);
                }
            };
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle()> MatrixFrameOptionsCommonElementStyle::empty_style = []() -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {};
        };

        ::std::function<MatrixFrameOptionsCommonElementStyle::ElementStyle(int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color)> MatrixFrameOptionsCommonElementStyle::mark_sub_matrix = [](int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color) -> MatrixFrameOptionsCommonElementStyle::ElementStyle {
            return [left_up_row_id, left_up_col_id, right_bottom_row_id, right_bottom_col_id, color](Element &ele, int x, int y, Element &separator_right, Element &separator_bottom, Element &separator_cross) {
                if (x >= left_up_col_id && x <= right_bottom_col_id && y >= left_up_row_id && y <= right_bottom_row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                    ele |= ::ftxui::bgcolor(color);
                }
                // left border
                if (x == left_up_col_id - 1 && y >= left_up_row_id && y <= right_bottom_row_id) {
                    separator_right |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                // top border
                if (y == left_up_row_id - 1 && x >= left_up_col_id && x <= right_bottom_col_id) {
                    separator_bottom |= ::ftxui::bgcolor(color);
                    separator_cross |= ::ftxui::bgcolor(color);
                }
                // left-top corss
                if (x == left_up_col_id - 1 && y == left_up_row_id - 1) {
                    separator_cross |= ::ftxui::bgcolor(color);
                }
            };
        };

        Component MatrixFrame(float* ptr, int rows, int cols, MatrixFrameOptions<float> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
          
            return Make<MatrixFrameBase<float>>(options);
        }

        Component MatrixFrame(double* ptr, int rows, int cols, MatrixFrameOptions<double> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
          
            return Make<MatrixFrameBase<double>>(options);
        }

        Component MatrixFrame(int* ptr, int rows, int cols, MatrixFrameOptions<int> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
          
            return Make<MatrixFrameBase<int>>(options);
        }

        #ifdef __CUDA__
        Component MatrixFrame(half* ptr, int rows, int cols, MatrixFrameOptions<half> options) {
            options.cols = cols;
            options.rows = rows;
            options.ptr = ptr;
          
            return Make<MatrixFrameBase<half>>(options);
        }
        #endif
        

        
       
    }
}

tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle operator|( tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle lhs, tui::component::MatrixFrameOptionsCommonElementStyle::ElementStyle rhs)
{
    return [lhs, rhs](ftxui::Element &ele, int x, int y, ftxui::Element &separator_right, ftxui::Element &separator_bottom, ftxui::Element &separator_cross) {
        lhs(ele, x, y, separator_right, separator_bottom, separator_cross);
        rhs(ele, x, y, separator_right, separator_bottom, separator_cross);
    };
}