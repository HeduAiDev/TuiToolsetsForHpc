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
            
            for (int i = this -> buffer_x_(); i < std::min(this -> cols, this ->buffer_x_() + this -> buffer_cols_); i ++) {
                Color font_color = Color::Gold3Bis;
                Color bg_color = Color::Grey3;
                Color separator_color = Color::Gold3;
                Color separator_bg_color = Color::Grey3;
                if (this -> col_label_style_map.count(i)) {
                    font_color = this -> col_label_style_map[i].color;
                    bg_color = this -> col_label_style_map[i].bgcolor;
                    separator_color = this -> col_label_style_map[i].separator_color;
                    separator_bg_color = this -> col_label_style_map[i].separator_bgcolor;
                }
                col_labels_arr.push_back(text(::std::to_string(i)) | center | frame | size(WIDTH, EQUAL, this -> text_width) | color(font_color) | bgcolor(bg_color));
                col_labels_arr.push_back(separator() | color(separator_color) | bgcolor(separator_bg_color));
            }
            return gridbox({col_labels_arr});
        }

        template<typename T>
        Element MatrixFrameBase<T>::getRowLabels() {
            ::std::vector<::std::vector<Element>> row_labels_arr;
            
            for (int i = this -> buffer_y_(); i < std::min(this ->rows - 1, this ->buffer_y_() + this -> buffer_rows_ - 1); i ++) {
                Color font_color = Color::Gold3Bis;
                Color bg_color = Color::Grey3;
                Color separator_color = Color::Gold3;
                Color separator_bg_color = Color::Grey3;
                if (this -> row_label_style_map.count(i)) {
                    font_color = this -> row_label_style_map[i].color;
                    bg_color = this -> row_label_style_map[i].bgcolor;
                    separator_color = this -> row_label_style_map[i].separator_color;
                    separator_bg_color = this -> row_label_style_map[i].separator_bgcolor;
                }
                row_labels_arr.push_back({text(::std::to_string(i)) | size(HEIGHT, EQUAL, 1) | center | color(font_color) | bgcolor(bg_color)});
                row_labels_arr.push_back({separator() | color(separator_color) | bgcolor(separator_bg_color)});
            }
            int i = std::min(this -> rows - 1, this ->buffer_y_() + this -> buffer_rows_ - 1);
            Color font_color = Color::Gold3Bis;
            Color bg_color = Color::Grey3;
            if (this -> row_label_style_map.count(i)) {
                font_color = this -> row_label_style_map[i].color;
                bg_color = this -> row_label_style_map[i].bgcolor;
            }
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
                ::std::vector<Element> empty_vector;
                ::std::vector<Element> &_separator_arr_last = _rows_arr.size() > 0 ? _rows_arr.back() : empty_vector;
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
                    Element separator_cross_bottom_right = separator();
                    _cols_arr.push_back(ele);
                    _cols_arr.push_back(separator_right);
                    _separator_arr.push_back(separator_bottom);
                    _separator_arr.push_back(separator_cross_bottom_right);
                }
                _rows_arr.push_back(_cols_arr);
                if (i != std::min(this -> rows -1, this -> buffer_y_() + this -> buffer_rows_ - 1)) {
                    _rows_arr.push_back(_separator_arr);
                }
            }
            for (int i = this -> buffer_y_(); i < std::min(this -> rows, this -> buffer_y_() + this -> buffer_rows_); i++) {
                for (int j = this -> buffer_x_(); j < std::min(this -> cols, this -> buffer_x_() + this -> buffer_cols_); j++) {
                    if (this -> point_style_map.count({j, i})) {
                        this -> point_style_map[{j, i}](2 * (j - this -> buffer_x_()), 2 * (i - this -> buffer_y_()), _rows_arr);
                    }
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

        using ElementStyle = MatrixFrameOptionsCommonElementStyle::ElementStyle;
        using CommonElementStyle = MatrixFrameOptionsCommonElementStyle;

        ::std::function<ElementStyle(int row_id, Color color)>
        CommonElementStyle::hight_light_row = [](int row_id, Color color) -> ElementStyle {
            return [row_id, color](int x, int y, std::vector<Elements> &elements) {
            };
        };

        ::std::function<ElementStyle(int col_id, Color color)>
        CommonElementStyle::hight_light_col = [](int col_id, Color color) -> ElementStyle {
            return [col_id, color](int x, int y, std::vector<Elements> &elements) {
            };
        };

        ::std::function<ElementStyle(int row_id, int col_id, Color color)>
        CommonElementStyle::mark_point = [](int row_id, int col_id, Color color) -> ElementStyle {
            return [row_id, col_id, color](int x, int y, std::vector<Elements> &elements) {
                Element* ele                            = utils::row_major_vect_get(elements, x, y);
                Element* separator_right                = utils::row_major_vect_get(elements, x + 1, y);
                Element* separator_left                 = utils::row_major_vect_get(elements, x - 1, y);
                Element* separator_top                  = utils::row_major_vect_get(elements, x, y - 1);
                Element* separator_bottom               = utils::row_major_vect_get(elements, x, y + 1);
                Element* separator_cross_top_right      = utils::row_major_vect_get(elements, x + 1, y - 1);
                Element* separator_cross_bottom_right   = utils::row_major_vect_get(elements, x + 1, y + 1);
                Element* separator_cross_top_left       = utils::row_major_vect_get(elements, x - 1, y - 1);
                Element* separator_cross_bottom_left    = utils::row_major_vect_get(elements, x - 1, y + 1);

                if (ele) *ele |= ::ftxui::bgcolor(color);
                if (separator_right) *separator_right |= ::ftxui::bgcolor(color);
                if (separator_left) *separator_left |= ::ftxui::bgcolor(color);
                if (separator_top) *separator_top |= ::ftxui::bgcolor(color);
                if (separator_bottom) *separator_bottom |= ::ftxui::bgcolor(color);
                if (separator_cross_top_right) *separator_cross_top_right |= ::ftxui::bgcolor(color);
                if (separator_cross_bottom_right) *separator_cross_bottom_right |= ::ftxui::bgcolor(color);
                if (separator_cross_top_left) *separator_cross_top_left |= ::ftxui::bgcolor(color);
                if (separator_cross_bottom_left) *separator_cross_bottom_left |= ::ftxui::bgcolor(color);

            };
        };

        ::std::function<ElementStyle(utils::pair_map<ElementStyle> &point_style_map, int row_id, int col_id, int rows, int cols)>
        CommonElementStyle::mark_point_trace = [](utils::pair_map<ElementStyle> &point_style_map, int row_id, int col_id, int rows, int cols ) -> ElementStyle {
            Color trace_color = Color::HSV(5, 222, 227);
            Color point_color = Color::HSV(5, 222, 227);
            std::function<ElementStyle(int)> row_trace = [trace_color, col_id, cols](int cur_col_id) -> ElementStyle {
                return [trace_color, tar_col_id = col_id, cur_col_id, cols] (int x, int y, std::vector<Elements> &elements) {
                    Element* separator_top                  = utils::row_major_vect_get(elements, x, y - 1);
                    Element* separator_bottom               = utils::row_major_vect_get(elements, x, y + 1);
                    Element* separator_cross_top_right      = utils::row_major_vect_get(elements, x + 1, y - 1);
                    Element* separator_cross_bottom_right   = utils::row_major_vect_get(elements, x + 1, y + 1);
                    // light = factor / (factor + x)
                    float factor = 10 + 60 * std::min(1.0, cols / 1000.0);
                    float light = std::max( 0.3f , factor / (factor + std::abs(cur_col_id - tar_col_id)));
                    if (separator_top) *separator_top |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_bottom) *separator_bottom |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_cross_bottom_right) *separator_cross_bottom_right |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_cross_top_right) *separator_cross_top_right |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                };
            };
            

            std::function<ElementStyle(int)> col_trace = [trace_color, row_id, rows](int cur_row_id) -> ElementStyle {
                return [trace_color, tar_row_id = row_id, cur_row_id, rows](int x, int y, std::vector<Elements> &elements) {
                    Element* separator_right                = utils::row_major_vect_get(elements, x + 1, y);
                    Element* separator_left                 = utils::row_major_vect_get(elements, x - 1, y);
                    Element* separator_cross_bottom_left    = utils::row_major_vect_get(elements, x - 1, y + 1);
                    Element* separator_cross_bottom_right   = utils::row_major_vect_get(elements, x + 1, y + 1);
                    // light = factor / (factor + x)
                    float factor = 10 + 60 * std::min(1.0, rows / 1000.0);
                    float light = std::max( 0.35f , factor / (factor + std::abs(cur_row_id - tar_row_id)));
                    if (separator_right) *separator_right |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_left) *separator_left |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_cross_bottom_right) *separator_cross_bottom_right |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                    if (separator_cross_bottom_left) *separator_cross_bottom_left |= ::ftxui::bgcolor(Color::HSV(5, 222, 227 * light));
                };
            };
            for (int row = 0; row < rows; row++) {
                if (point_style_map.count({col_id, row})) {
                    point_style_map[{col_id, row}] = col_trace(row) | point_style_map[{col_id, row}];
                } else {
                    point_style_map[{col_id, row}] = col_trace(row);
                }
            }
            for (int col = 0; col < cols; col++) {
                if (point_style_map.count({col, row_id})) {
                    point_style_map[{col, row_id}] = row_trace(col) | point_style_map[{col, row_id}];
                } else {
                    point_style_map[{col, row_id}] = row_trace(col);
                }
            }
            return [row_id, col_id, trace_color, point_color](int x, int y, std::vector<Elements> &elements) {
                Element* ele                            = utils::row_major_vect_get(elements, x, y);
                Element* separator_right                = utils::row_major_vect_get(elements, x + 1, y);
                Element* separator_left                 = utils::row_major_vect_get(elements, x - 1, y);
                Element* separator_top                  = utils::row_major_vect_get(elements, x, y - 1);
                Element* separator_bottom               = utils::row_major_vect_get(elements, x, y + 1);
                Element* separator_cross_top_right      = utils::row_major_vect_get(elements, x + 1, y - 1);
                Element* separator_cross_bottom_right   = utils::row_major_vect_get(elements, x + 1, y + 1);
                Element* separator_cross_top_left       = utils::row_major_vect_get(elements, x - 1, y - 1);
                Element* separator_cross_bottom_left    = utils::row_major_vect_get(elements, x - 1, y + 1);

                if (ele) *ele |= ::ftxui::bgcolor(point_color);
                if (separator_right) *separator_right |= ::ftxui::bgcolor(point_color);
                if (separator_left) *separator_left |= ::ftxui::bgcolor(point_color);
                if (separator_top) *separator_top |= ::ftxui::bgcolor(point_color);
                if (separator_bottom) *separator_bottom |= ::ftxui::bgcolor(point_color);
                if (separator_cross_top_right) *separator_cross_top_right |= ::ftxui::bgcolor(point_color);
                if (separator_cross_bottom_right) *separator_cross_bottom_right |= ::ftxui::bgcolor(point_color);
                if (separator_cross_top_left) *separator_cross_top_left |= ::ftxui::bgcolor(point_color);
                if (separator_cross_bottom_left) *separator_cross_bottom_left |= ::ftxui::bgcolor(point_color);
            };
        };

        ::std::function<ElementStyle()> CommonElementStyle::empty_style = []() -> ElementStyle {
            return [](int x, int y, std::vector<Elements> &elements) {};
        };

        ::std::function<ElementStyle(int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color)> CommonElementStyle::mark_sub_matrix = [](int left_up_row_id, int left_up_col_id, int right_bottom_row_id, int right_bottom_col_id, Color color) -> ElementStyle {
            return [left_up_row_id, left_up_col_id, right_bottom_row_id, right_bottom_col_id, color](int x, int y, std::vector<Elements> &elements) {
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
    return [lhs, rhs](int x, int y, std::vector<ftxui::Elements> &elements) {
        lhs(x, y, elements);
        rhs(x, y, elements);
    };
}